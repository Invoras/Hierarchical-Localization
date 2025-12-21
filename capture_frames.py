#!/usr/bin/env python3
"""
Real-Time Drone Localization and Visualization

Connects to a LiveKit drone stream and:
1. Localizes the drone position on a sparse 3D map every 5 seconds
2. Displays real-time 3D visualization in a browser at http://127.0.0.1:8050

The visualization shows:
- Full 3D sparse reconstruction (reference map)
- Drone camera position and orientation (green frustum)
- Position (X,Y,Z) and orientation (yaw/pitch/roll) information
- Localization quality (inlier count)

Uses pre-loaded ML models (NetVLAD, SuperPoint, LightGlue) for fast localization.
"""

import asyncio
import logging
import os
import sys
import threading
import copy
from datetime import datetime
from pathlib import Path
import requests
import cv2
import numpy as np
from livekit import rtc
import torch
import h5py
import pycolmap
from scipy.spatial.transform import Rotation
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.utils.io import read_image
from hloc.utils import viz_3d
from hloc.extractors.netvlad import NetVLAD
from hloc.extractors.superpoint import SuperPoint
from hloc.matchers.lightglue import LightGlue

# Configuration
ROOM_NAME = "drone-stream"
TOKEN_SERVER_URL = "https://n189ebyfq6.execute-api.eu-central-1.amazonaws.com/default/token"
FRAME_INTERVAL = 0.1  # seconds
WARMUP_DELAY = 10  # seconds - wait for SFU bandwidth estimation to ramp up to full quality

# Localization configuration
LOCALIZE_INTERVAL = 5.0  # seconds - localize drone position every 5 seconds
DASH_PORT = 8050
DASH_HOST = "127.0.0.1"
DASH_REFRESH_INTERVAL = 2000  # milliseconds - browser refresh rate (increased to reduce load)

# Third-person camera configuration
CAMERA_OFFSET_BACK = 1.5    # Meters behind the drone
CAMERA_OFFSET_UP = 0.0      # Meters above the drone
CAMERA_LOOK_AHEAD = 0.0     # Look directly at drone (0.0 = no offset)

# Default paths for localization
SPARSE_MODEL_PATH = "final_inputs/v1/sparse/0"
REFERENCE_FEATURES_PATH = "outputs/sfm_1s/feats-superpoint-n4096-r1024.h5"
REFERENCE_IMAGES_PATH = "final_inputs/v1/images"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_netvlad_direct(model, image_path, retrieval_conf, device):
    """Extract NetVLAD global descriptor using pre-loaded model."""

    # Load and preprocess image (NetVLAD uses RGB, not grayscale)
    image = read_image(image_path, grayscale=False)
    original_size = np.array(image.shape[:2])

    # Resize if needed
    if "resize_max" in retrieval_conf["preprocessing"]:
        h, w = image.shape[:2]
        max_size = retrieval_conf["preprocessing"]["resize_max"]
        scale = max_size / max(h, w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert to tensor (RGB, so 3 channels)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)[None].to(device) / 255.0

    # Extract global descriptor
    with torch.no_grad():
        pred = model({"image": image_tensor})

    # Convert to numpy
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    pred["image_size"] = original_size

    return pred


def extract_superpoint_direct(model, image_path, feature_conf, device):
    """Extract SuperPoint features using pre-loaded model (bypasses extract_features.main())."""

    # Load and preprocess image
    image = read_image(image_path, grayscale=feature_conf["preprocessing"]["grayscale"])
    original_size = np.array(image.shape[:2])

    # Resize if needed
    if "resize_max" in feature_conf["preprocessing"]:
        h, w = image.shape[:2]
        max_size = feature_conf["preprocessing"]["resize_max"]
        scale = max_size / max(h, w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert to tensor
    image_tensor = torch.from_numpy(image).float()[None, None].to(device) / 255.0

    # Extract features
    with torch.no_grad():
        pred = model({"image": image_tensor})

    # Convert to numpy
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

    # Scale keypoints back to original resolution
    if "keypoints" in pred:
        size = np.array(image_tensor.shape[-2:][::-1])
        scales = (original_size[::-1] / size).astype(np.float32)
        pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
        if "scales" in pred:
            pred["scales"] *= scales.mean()

    pred["image_size"] = original_size

    return pred


def match_features_direct(model, features0, features1, device):
    """Match features using pre-loaded model (bypasses match_features.main())."""

    # Prepare data for matching (LightGlue expects image tensors even though it only uses their size)
    data = {
        "keypoints0": torch.from_numpy(features0["keypoints"]).float()[None].to(device),
        "keypoints1": torch.from_numpy(features1["keypoints"]).float()[None].to(device),
        "descriptors0": torch.from_numpy(features0["descriptors"]).float()[None].to(device),
        "descriptors1": torch.from_numpy(features1["descriptors"]).float()[None].to(device),
        "image0": torch.empty((1, 1) + tuple(features0["image_size"][::-1])).to(device),
        "image1": torch.empty((1, 1) + tuple(features1["image_size"][::-1])).to(device),
    }

    # Match
    with torch.no_grad():
        pred = model(data)

    # Convert to numpy
    matches = pred["matches0"][0].cpu().numpy()
    scores = pred["matching_scores0"][0].cpu().numpy()

    return matches, scores


class LocalizationState:
    """Thread-safe shared state for localization results"""

    def __init__(self, sparse_model):
        self._lock = threading.Lock()
        self._latest_pose = None  # Camera pose (Rigid3d)
        self._latest_camera = None  # Camera params
        self._latest_inliers = None  # Inlier count
        self._latest_total_matches = None  # Total matches
        self._latest_timestamp = None  # When localized
        self._latest_frame_num = None  # Frame number
        self._error_message = None  # Error if localization failed
        self.sparse_model = sparse_model  # Reference reconstruction (immutable)

    def update_pose(self, pose_data):
        """Update with new localization result"""
        with self._lock:
            self._latest_pose = pose_data['cam_from_world']
            self._latest_camera = pose_data['camera']
            self._latest_inliers = pose_data['inliers']
            self._latest_total_matches = pose_data['total_matches']
            self._latest_frame_num = pose_data['frame_num']
            self._latest_timestamp = pose_data['timestamp']
            self._error_message = None  # Clear error on success

    def set_error(self, error_msg):
        """Set error message (keeps last good pose)"""
        with self._lock:
            self._error_message = error_msg

    def get_latest(self):
        """Get latest localization data (thread-safe copy)"""
        with self._lock:
            if self._latest_pose is None:
                return None
            return {
                'cam_from_world': self._latest_pose,
                'camera': self._latest_camera,
                'inliers': self._latest_inliers,
                'total_matches': self._latest_total_matches,
                'frame_num': self._latest_frame_num,
                'timestamp': self._latest_timestamp,
                'error': self._error_message
            }


def calculate_third_person_camera(
    drone_world_pos: np.ndarray,
    drone_rotation_matrix: np.ndarray,
    sparse_model: pycolmap.Reconstruction,
    offset_back: float = 3.0,
    offset_up: float = 1.5,
    look_ahead: float = 0.0
) -> dict:
    """
    Calculate third-person camera position for Plotly visualization.

    Args:
        drone_world_pos: [x, y, z] drone position in world coordinates
        drone_rotation_matrix: 3x3 rotation from camera to world (c2w)
        sparse_model: Reconstruction to get data bounds
        offset_back: Meters behind the drone (default: 3.0)
        offset_up: Meters above the drone (default: 1.5)
        look_ahead: Meters ahead of drone to look at (default: 0.0)

    Returns:
        Dictionary with 'eye', 'center', 'up' for Plotly scene_camera
    """
    # Extract orientation vectors from rotation matrix
    # Columns represent camera axes in world frame
    right_world = drone_rotation_matrix[:, 0]
    down_world = drone_rotation_matrix[:, 1]   # Y-down convention
    forward_world = drone_rotation_matrix[:, 2]  # Camera +Z direction

    # Calculate camera position: behind and above the drone
    camera_pos_world = (
        drone_world_pos
        - forward_world * offset_back   # Move opposite to forward
        - down_world * offset_up         # Move opposite to down (i.e., up)
    )

    # Camera looks at drone position (or slightly ahead)
    look_at_world = drone_world_pos + forward_world * look_ahead

    # Get data bounds for coordinate transformation
    # Calculate bounds directly from 3D points (more compatible across pycolmap versions)
    all_points = np.array([p3D.xyz for p3D in sparse_model.points3D.values()])

    # Use percentiles to filter outliers (similar to compute_bounding_box)
    data_min = np.percentile(all_points, 0.1, axis=0)
    data_max = np.percentile(all_points, 99.9, axis=0)
    data_center = (data_min + data_max) / 2.0
    data_extent = data_max - data_min

    # Convert to Plotly domain coordinates
    max_extent = np.max(data_extent)
    scale_factor = 2.0 / max_extent

    camera_domain = (camera_pos_world - data_center) * scale_factor
    target_domain = (look_at_world - data_center) * scale_factor

    # Up vector: opposite of down direction
    up_world = -down_world

    return {
        'eye': {
            'x': float(camera_domain[0]),
            'y': float(camera_domain[1]),
            'z': float(camera_domain[2])
        },
        'center': {
            'x': float(target_domain[0]),
            'y': float(target_domain[1]),
            'z': float(target_domain[2])
        },
        'up': {
            'x': float(up_world[0]),
            'y': float(up_world[1]),
            'z': float(up_world[2])
        }
    }


def create_dash_app(localization_state):
    """Create Dash app for real-time 3D visualization"""
    app = Dash(__name__)

    # Pre-render the sparse reconstruction ONCE (cached)
    logger.info("Pre-rendering sparse reconstruction for visualization...")
    base_fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(
        base_fig,
        localization_state.sparse_model,
        color='rgba(255,0,0,0.0)',
        points_rgb=True,
        cameras=False,
        min_track_length=3,
        name="Reference Map"
    )
    # Preserve camera view across updates
    base_fig.update_layout(uirevision='constant')
    logger.info("Sparse reconstruction pre-rendered!")

    app.layout = html.Div([
        html.H1("Real-Time Drone Localization", style={'textAlign': 'center'}),

        # Status display
        html.Div(id='status-text', style={'padding': '20px', 'fontSize': '14px'}),

        # Camera mode toggle
        html.Div([
            html.Label("Camera Mode:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.RadioItems(
                id='camera-mode-toggle',
                options=[
                    {'label': ' Auto-Follow Drone', 'value': 'auto'},
                    {'label': ' Free Camera (Manual Control)', 'value': 'manual'}
                ],
                value='auto',  # Default to auto-follow
                labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                style={'display': 'inline-block'}
            )
        ], style={'padding': '10px 20px', 'backgroundColor': '#f0f0f0', 'marginBottom': '10px'}),

        # 3D visualization
        dcc.Graph(
            id='3d-plot',
            style={'height': '80vh'}
        ),

        # Auto-refresh interval (every 1 second)
        dcc.Interval(
            id='interval-component',
            interval=DASH_REFRESH_INTERVAL,  # milliseconds
            n_intervals=0
        )
    ])

    @app.callback(
        [Output('3d-plot', 'figure'),
         Output('status-text', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('camera-mode-toggle', 'value')]
    )
    def update_visualization(n, camera_mode):
        """Update 3D visualization with latest localization data"""
        # Get latest localization data from shared state
        latest = localization_state.get_latest()
        logger.info(f"Dash callback: latest={latest is not None}")

        if latest is None:
            # No localization yet - show base figure with waiting message
            logger.info("  No localization data yet")
            return base_fig, html.Div("Waiting for first localization...", style={'color': 'gray', 'fontSize': '16px'})

        try:
            logger.info(f"  Rendering visualization for frame {latest['frame_num']}")

            # Clone the pre-rendered base figure (sparse reconstruction)
            fig = go.Figure(base_fig)

            # Add drone camera frustum (green) to the existing figure
            logger.info("  Adding camera frustum...")
            viz_3d.plot_camera_colmap(
                fig,
                latest['cam_from_world'],
                latest['camera'],
                color="rgba(0,255,0,0.6)",
                name="Drone Camera",
                fill=True,
                text=f"Frame: {latest['frame_num']}\nInliers: {latest['inliers']}"
            )

            # Get drone position and orientation (used for camera and status text)
            c2w = latest['cam_from_world'].inverse()
            drone_pos = c2w.translation
            drone_rot = c2w.rotation.matrix()

            # Calculate and apply third-person camera if in auto-follow mode
            if camera_mode == 'auto':
                logger.info("  Calculating third-person camera position...")
                camera_config = calculate_third_person_camera(
                    drone_pos,
                    drone_rot,
                    localization_state.sparse_model,
                    offset_back=CAMERA_OFFSET_BACK,
                    offset_up=CAMERA_OFFSET_UP,
                    look_ahead=CAMERA_LOOK_AHEAD
                )

                # Apply camera update with changing uirevision to force camera update
                fig.update_layout(
                    scene_camera=camera_config,
                    uirevision=n  # Change uirevision to force camera update
                )
            else:
                # Manual mode: preserve user camera control
                fig.update_layout(uirevision='constant')

            # 3. Build status text
            logger.info("  Building status text...")
            pos = drone_pos
            rot = Rotation.from_matrix(drone_rot)
            yaw, pitch, roll = rot.as_euler('zyx', degrees=True)

            status_children = [
                html.H3("Drone Position (meters)"),
                html.P(f"X: {pos[0]:.3f} | Y: {pos[1]:.3f} | Z: {pos[2]:.3f}"),
                html.H3("Orientation (degrees)"),
                html.P(f"Yaw: {yaw:.1f}° | Pitch: {pitch:.1f}° | Roll: {roll:.1f}°"),
                html.H3("Localization Info"),
                html.P(f"Frame: {latest['frame_num']} | Inliers: {latest['inliers']}/{latest['total_matches']} | Time: {latest['timestamp'].strftime('%H:%M:%S')}")
            ]

            # Add error message if present
            if latest['error']:
                status_children.append(
                    html.Div(
                        f"⚠ Warning: {latest['error']}",
                        style={'color': 'red', 'fontWeight': 'bold', 'marginTop': '10px'}
                    )
                )

            logger.info("  Visualization rendered successfully")
            return fig, html.Div(status_children)
        except Exception as e:
            logger.error(f"  Error rendering visualization: {e}")
            import traceback
            traceback.print_exc()
            empty_fig = viz_3d.init_figure()
            error_msg = html.Div([
                html.H3("Error rendering visualization", style={'color': 'red'}),
                html.P(str(e))
            ])
            return empty_fig, error_msg

    return app


def run_dash_server(app):
    """Run Dash server in background thread"""
    try:
        app.run(debug=False, host=DASH_HOST, port=DASH_PORT, use_reloader=False)
    except Exception as e:
        logger.error(f"Dash server error: {e}")


class FrameCapture:
    def __init__(self, output_folder: str, sparse_model_path: str, reference_features_path: str, reference_images_path: str):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0
        self.last_localize_time = 0
        self.localizing = False
        self.running = True

        logger.info(f"Output folder: {self.output_folder}")

        # Store paths
        self.sparse_model_path = Path(sparse_model_path)
        self.reference_features_path = Path(reference_features_path)
        self.reference_images_path = Path(reference_images_path)

        # Load sparse model
        logger.info("Loading sparse reconstruction model...")
        self.sparse_model = pycolmap.Reconstruction(self.sparse_model_path)
        self.ref_image_list = sorted([img.name for img in self.sparse_model.images.values()])
        logger.info(f"  Loaded model: {len(self.sparse_model.images)} images, {len(self.sparse_model.points3D)} 3D points")

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"  Using device: {self.device}")

        # Feature extraction configs
        self.retrieval_conf = extract_features.confs["netvlad"]
        self.feature_conf = extract_features.confs["superpoint_aachen"]
        self.matcher_conf = match_features.confs["superpoint+lightglue"]

        # Load ML models
        logger.info("  Loading NetVLAD, SuperPoint, and LightGlue models...")
        self.netvlad_model = NetVLAD(self.retrieval_conf['model']).eval().to(self.device)
        self.superpoint_model = SuperPoint(self.feature_conf['model']).eval().to(self.device)
        self.lightglue_model = LightGlue(self.matcher_conf['model']).eval().to(self.device)
        logger.info("  Models loaded successfully!")

        # Extract NetVLAD features for reference images (if needed)
        # Use a permanent cache location (not in timestamped output folder)
        cache_dir = Path("outputs/localization_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.global_features_ref = cache_dir / "global-feats-netvlad.h5"

        if not self.global_features_ref.exists():
            logger.info("  Extracting NetVLAD features for reference images (one-time setup)...")
            extract_features.main(
                self.retrieval_conf,
                self.reference_images_path,
                image_list=self.ref_image_list,
                feature_path=self.global_features_ref,
                overwrite=False
            )
            logger.info("  NetVLAD cache built successfully!")
        else:
            logger.info("  Using cached NetVLAD features for reference images")

        # Create localizer
        self.loc_config = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True}
        }
        self.localizer = QueryLocalizer(self.sparse_model, self.loc_config)

        # Initialize shared state for visualization
        self.localization_state = LocalizationState(self.sparse_model)

        # Start Dash server in background thread
        logger.info("  Starting Dash visualization server...")
        dash_app = create_dash_app(self.localization_state)
        dash_thread = threading.Thread(
            target=run_dash_server,
            args=(dash_app,),
            daemon=True
        )
        dash_thread.start()
        logger.info(f"📊 Visualization available at http://{DASH_HOST}:{DASH_PORT}")

    async def connect_to_room(self):
        """Fetch token and connect to LiveKit room"""
        try:
            # Fetch token from AWS Lambda
            logger.info(f"Fetching token from {TOKEN_SERVER_URL}...")
            response = requests.post(
                TOKEN_SERVER_URL,
                json={
                    "roomName": ROOM_NAME,
                    "participantName": "frame-capture-client",
                    "isPublisher": False
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            url = data["url"]
            token = data["token"]

            logger.info(f"Connecting to room: {ROOM_NAME}")

            # Connect to LiveKit room
            room = rtc.Room()

            # Set up event handlers
            @room.on("track_subscribed")
            def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
                logger.info(f"Track subscribed: {track.kind} from {participant.identity}")

                if track.kind == rtc.TrackKind.KIND_VIDEO:
                    # Log detailed track information
                    logger.info(f"📺 Video Track Details:")
                    logger.info(f"   Publication resolution: {publication.width}x{publication.height}")
                    logger.info(f"   Track SID: {publication.sid}")
                    logger.info(f"   Track Name: {publication.name}")
                    logger.info(f"   Simulcasted: {publication.simulcasted}")
                    logger.info(f"   MIME Type: {publication.mime_type}")

                    asyncio.create_task(self.process_video_track(track, publication))

            @room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                logger.info(f"Participant connected: {participant.identity}")

            @room.on("participant_disconnected")
            def on_participant_disconnected(participant: rtc.RemoteParticipant):
                logger.info(f"Participant disconnected: {participant.identity}")

            # Connect with room options
            # Note: adaptive_stream is disabled by default in Python SDK when not using UI elements
            room_options = rtc.RoomOptions(
                auto_subscribe=True,  # Automatically subscribe to tracks
                dynacast=False,  # Disable dynamic quality - we want full resolution always
            )
            await room.connect(url, token, options=room_options)
            logger.info("✅ Connected to LiveKit room!")
            logger.info("   (Dynacast disabled - requesting full quality from start)")

            return room

        except requests.RequestException as e:
            logger.error(f"Failed to fetch token: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to room: {e}")
            raise

    async def process_video_track(self, track: rtc.VideoTrack, publication: rtc.TrackPublication):
        """Process incoming video frames"""
        logger.info("🎥 Video track received, starting warm-up period...")

        video_stream = rtc.VideoStream(track)
        first_frame = True
        warmup_complete = False
        warmup_start_time = None
        last_countdown_log = -1

        async for frame_event in video_stream:
            if not self.running:
                break

            # Log first frame details
            if first_frame:
                frame = frame_event.frame
                logger.info(f"🎬 First frame received:")
                logger.info(f"   Frame resolution: {frame.width}x{frame.height}")
                logger.info(f"   Publication resolution: {publication.width}x{publication.height}")
                logger.info(f"")
                logger.info(f"⏳ Warming up connection for {WARMUP_DELAY}s to allow SFU quality ramp-up...")
                warmup_start_time = asyncio.get_event_loop().time()
                first_frame = False

            # Warm-up period: wait for SFU to ramp up quality
            if not warmup_complete:
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - warmup_start_time

                # Log countdown every 5 seconds (avoid duplicate logs)
                elapsed_rounded = int(elapsed)
                if elapsed_rounded % 5 == 0 and elapsed_rounded > 0 and elapsed_rounded != last_countdown_log:
                    remaining = WARMUP_DELAY - elapsed
                    if remaining > 0:
                        logger.info(f"   Warm-up: {int(remaining)}s remaining...")
                        last_countdown_log = elapsed_rounded

                # Check if warm-up is complete
                if elapsed >= WARMUP_DELAY:
                    frame = frame_event.frame
                    logger.info(f"")
                    logger.info(f"✅ Warm-up complete! Final resolution check:")
                    logger.info(f"   Frame resolution: {frame.width}x{frame.height}")

                    if frame.width != 1920 or frame.height != 1080:
                        logger.warning(f"⚠️  WARNING: Still receiving {frame.width}x{frame.height} instead of 1920x1080")
                        logger.warning(f"   SFU may need more time or connection bandwidth is limited")
                    else:
                        logger.info(f"✅ SUCCESS: Receiving full 1080p resolution!")

                    logger.info(f"")
                    logger.info(f"📍 Localizing drone position every {LOCALIZE_INTERVAL}s...")
                    warmup_complete = True
                    self.last_localize_time = asyncio.get_event_loop().time()
                continue

            # Localization after warm-up
            current_time = asyncio.get_event_loop().time()

            # Localize every LOCALIZE_INTERVAL (5 seconds)
            if current_time - self.last_localize_time >= LOCALIZE_INTERVAL:
                if not self.localizing:  # Prevent overlap
                    self.localizing = True
                    asyncio.create_task(self.localize_frame(frame_event.frame))
                    self.last_localize_time = current_time
                else:
                    logger.warning("Skipping localization - previous still running")

    async def save_frame(self, frame: rtc.VideoFrame):
        """Convert and save video frame as image"""
        try:
            # Convert to numpy array (I420 -> BGR)
            buffer = frame.convert(rtc.VideoBufferType.RGBA)

            # Get frame data as numpy array
            data = np.frombuffer(buffer.data, dtype=np.uint8)
            img = data.reshape((buffer.height, buffer.width, 4))

            # Convert RGBA to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            # Generate filename with frame number and timestamp
            timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]  # milliseconds
            filename = f"frame_{self.frame_count:06d}_{timestamp}.jpg"
            filepath = self.output_folder / filename

            # Save image
            cv2.imwrite(str(filepath), img_bgr)

            self.frame_count += 1
            if self.frame_count % 10 == 0:
            	logger.info(f"💾 Saved frame {self.frame_count}: {filename}")

        except Exception as e:
            logger.error(f"Failed to save frame: {e}")

    def frame_to_numpy(self, frame: rtc.VideoFrame) -> np.ndarray:
        """Convert LiveKit frame to RGB numpy array for localization"""
        buffer = frame.convert(rtc.VideoBufferType.RGBA)
        data = np.frombuffer(buffer.data, dtype=np.uint8)
        img = data.reshape((buffer.height, buffer.width, 4))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # RGB for NetVLAD
        return img_rgb

    async def localize_frame(self, frame: rtc.VideoFrame):
        """Localize current frame using pre-loaded models"""
        try:
            # Increment frame count
            self.frame_count += 1

            # Convert frame to numpy
            img_array = self.frame_to_numpy(frame)

            # Save as temp image for processing
            query_name = f"query_frame_{self.frame_count}.jpg"
            query_path = self.output_folder / query_name
            cv2.imwrite(str(query_path), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

            # 1. Extract NetVLAD features
            netvlad_features = extract_netvlad_direct(
                self.netvlad_model, query_path, self.retrieval_conf, self.device
            )

            # Save to H5 for retrieval
            global_features_query = self.output_folder / "global-feats-query.h5"
            with h5py.File(str(global_features_query), "w") as fd:
                grp = fd.create_group(query_name)
                for k, v in netvlad_features.items():
                    grp.create_dataset(k, data=v)

            # 2. Image retrieval (top 10)
            retrieval_pairs = self.output_folder / "pairs-retrieval.txt"
            pairs_from_retrieval.main(
                global_features_query,
                retrieval_pairs,
                num_matched=10,
                db_descriptors=self.global_features_ref,
                db_model=self.sparse_model_path,
                query_prefix=None,
                query_list=[query_name]
            )

            # Get retrieved reference names
            retrieved_refs = []
            with open(retrieval_pairs, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        retrieved_refs.append(parts[1])

            # 3. Extract SuperPoint features
            query_features = extract_superpoint_direct(
                self.superpoint_model, query_path, self.feature_conf, self.device
            )

            # 4. Match with LightGlue
            matches_path = self.output_folder / "matches.h5"
            all_matches = {}

            with h5py.File(str(self.reference_features_path), "r") as ref_fd:
                for ref_name in retrieved_refs:
                    if ref_name not in ref_fd:
                        continue

                    ref_group = ref_fd[ref_name]
                    ref_features = {
                        "keypoints": ref_group["keypoints"][:],
                        "descriptors": ref_group["descriptors"][:],
                        "image_size": ref_group["image_size"][:],
                    }

                    matches, scores = match_features_direct(
                        self.lightglue_model,
                        query_features,
                        ref_features,
                        self.device
                    )

                    pair_key = f"{query_name}/{ref_name}"
                    all_matches[pair_key] = {
                        "matches0": matches,
                        "matching_scores0": scores,
                    }

            # Save matches
            with h5py.File(str(matches_path), "w") as fd:
                for pair_key, match_data in all_matches.items():
                    grp = fd.create_group(pair_key)
                    grp.create_dataset("matches0", data=match_data["matches0"])
                    grp.create_dataset("matching_scores0", data=match_data["matching_scores0"])

            # Save features
            local_features = self.output_folder / "feats-query.h5"
            with h5py.File(str(local_features), "w") as fd:
                with h5py.File(str(self.reference_features_path), "r") as ref_fd:
                    for ref_name in retrieved_refs:
                        if ref_name in ref_fd:
                            ref_fd.copy(ref_name, fd)

                q_grp = fd.create_group(query_name)
                for k, v in query_features.items():
                    q_grp.create_dataset(k, data=v)

            # 5. PnP+RANSAC localization
            ref_ids = [self.sparse_model.find_image_with_name(name).image_id
                       for name in retrieved_refs]
            camera = pycolmap.infer_camera_from_image(query_path)

            ret, log = pose_from_cluster(
                self.localizer,
                query_name,
                camera,
                ref_ids,
                local_features,
                matches_path
            )

            if ret is not None:
                # Success - update shared state
                self.localization_state.update_pose({
                    'cam_from_world': ret['cam_from_world'],
                    'camera': ret['camera'],
                    'inliers': ret['num_inliers'],
                    'total_matches': len(ret['inlier_mask']),
                    'frame_num': self.frame_count,
                    'timestamp': datetime.now()
                })
                logger.info(f"✓ Localized frame {self.frame_count}: "
                           f"{ret['num_inliers']} inliers")
                logger.info(f"  State updated - checking: {self.localization_state.get_latest() is not None}")
            else:
                # Failed
                self.localization_state.set_error(
                    f"Failed to localize frame {self.frame_count}"
                )
                logger.warning(f"✗ Localization failed for frame {self.frame_count}")

            # Cleanup temp file
            query_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Localization error: {e}")
            self.localization_state.set_error(str(e))
        finally:
            self.localizing = False

    async def run(self):
        """Main run loop"""
        room = None
        try:
            room = await self.connect_to_room()

            logger.info(f"📍 Localizing drone every {LOCALIZE_INTERVAL}s. Press Ctrl+C to stop.")

            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping capture...")
            self.running = False
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
        finally:
            if room:
                await room.disconnect()
            logger.info(f"✅ Localization complete! Output in {self.output_folder}")


async def main():
    """Entry point"""
    # Use a single temp folder (not timestamped) for localization temp files
    output_folder = "outputs/localization_temp"

    logger.info("=" * 60)
    logger.info("🚁 Drone Frame Capture + Real-Time Localization")
    logger.info("=" * 60)

    # Create and run capture with localization
    capture = FrameCapture(
        output_folder,
        sparse_model_path=SPARSE_MODEL_PATH,
        reference_features_path=REFERENCE_FEATURES_PATH,
        reference_images_path=REFERENCE_IMAGES_PATH
    )
    await capture.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
