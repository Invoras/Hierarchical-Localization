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
import json
from datetime import datetime
import time
from pathlib import Path
import requests
import cv2
import numpy as np
from livekit import rtc
import torch
import h5py
import pycolmap
from scipy.spatial.transform import Rotation
from dash import Dash, dcc, html, Input, Output, State, Patch, no_update
import plotly.graph_objects as go
from flask import jsonify, send_from_directory

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
LOCALIZE_INTERVAL = 20.0  # seconds - localize drone position every 20 seconds (changed from 5.0 for telemetry-based dead reckoning)
DASH_PORT = 8050
DASH_HOST = "127.0.0.1"
DASH_REFRESH_INTERVAL = 100  # milliseconds - browser refresh rate (increased to reduce load)

# Third-person camera configuration
CAMERA_OFFSET_BACK = 3.5    # Meters behind the drone
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


class HybridLocalizationState:
    """Thread-safe state for hybrid HLOC + telemetry localization

    Combines HLOC fixes (every 20s) with telemetry-based dead reckoning for
    continuous position updates.

    Coordinate System: HLOC world frame with Y-up convention
    - Y-axis: Vertical (altitude is position[1])
    - Horizontal plane: X-Z
    - Yaw: Rotation around Y-axis
    """

    def __init__(self, sparse_model):
        self._lock = threading.Lock()

        # HLOC state (ground truth, updated every 20s)
        self._latest_hloc_pose = None  # pycolmap.Rigid3d (cam_from_world)
        self._latest_hloc_camera = None  # Camera parameters
        self._latest_hloc_inliers = None  # Inlier count
        self._latest_hloc_matches = None  # Total matches
        self._latest_hloc_timestamp = None  # datetime when HLOC ran
        self._latest_hloc_frame = None  # Frame number
        self._hloc_initialized = False  # Flag for first HLOC success

        # Telemetry state (continuous updates)
        self._latest_telemetry = None  # Full telemetry dict
        self._last_telemetry_timestamp_ns = None  # For staleness detection
        self._last_telemetry_time = None  # For dt calculation

        # Dead reckoning state (updated continuously from telemetry)
        self._dr_position = None  # [x, y, z] in HLOC world frame (y is altitude!)
        self._dr_rotation_matrix = None  # 3x3 rotation matrix in HLOC frame

        # Calibration parameters (computed after first HLOC)
        self._heading_offset_deg = None  # NED→HLOC yaw offset
        self._altitude_offset_m = None  # Telemetry altitude to HLOC Y offset
        self._calibrated = False

        # Status tracking
        self._pose_source = "waiting"  # "hloc", "telemetry", "waiting"
        self._error_message = None

        # Reference data
        self.sparse_model = sparse_model

    def update_hloc_pose(self, pose_data):
        """Update with new HLOC localization result"""
        with self._lock:
            # Store HLOC data
            self._latest_hloc_pose = pose_data['cam_from_world']
            self._latest_hloc_camera = pose_data['camera']
            self._latest_hloc_inliers = pose_data['inliers']
            self._latest_hloc_matches = pose_data['total_matches']
            self._latest_hloc_frame = pose_data['frame_num']
            self._latest_hloc_timestamp = pose_data['timestamp']
            self._error_message = None

            # Extract position and orientation
            c2w = self._latest_hloc_pose.inverse()
            position = c2w.translation  # np.array [x, y, z]
            rotation = c2w.rotation.matrix()  # 3x3 numpy array

            # Reset dead reckoning to HLOC position (corrects drift)
            self._dr_position = np.array(position)
            self._dr_rotation_matrix = np.array(rotation)

            # Calibrate on first HLOC
            if not self._hloc_initialized:
                self._hloc_initialized = True
                logging.info("First HLOC fix received - system initialized")
                if self._latest_telemetry is not None:
                    self._calibrate()
            elif self._calibrated and self._latest_telemetry is not None:
                # Recalibrate on every HLOC to handle drift in offsets
                self._calibrate()

            self._pose_source = "hloc"
            logging.info(f"HLOC fix: position=[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}], inliers={self._latest_hloc_inliers}")

    def update_telemetry(self, telemetry_dict):
        """Process incoming telemetry data"""
        with self._lock:
            # Check for stale data (same timestamp_ns)
            current_timestamp_ns = telemetry_dict.get('timestamp_ns')
            if current_timestamp_ns == self._last_telemetry_timestamp_ns:
                return  # Skip stale data

            # Store previous telemetry for dt calculation
            prev_telemetry = self._latest_telemetry
            prev_time = self._last_telemetry_time

            # Update telemetry state
            self._latest_telemetry = telemetry_dict
            self._last_telemetry_timestamp_ns = current_timestamp_ns
            self._last_telemetry_time = datetime.now()

            # Calibrate if HLOC initialized but not yet calibrated
            if self._hloc_initialized and not self._calibrated:
                self._calibrate()

            # Perform dead reckoning if calibrated
            if self._calibrated and prev_telemetry is not None and prev_time is not None:
                self._update_dead_reckoning(prev_telemetry, telemetry_dict, prev_time)
                self._pose_source = "telemetry"

    def _calibrate(self):
        """Compute heading and altitude offsets (NED → HLOC transformation)"""
        if self._latest_telemetry is None or self._dr_rotation_matrix is None:
            logging.warning("Cannot calibrate: missing telemetry or HLOC data")
            return

        # Extract HLOC heading (yaw) from rotation matrix
        # Y-axis is up, yaw is rotation around Y in X-Z plane
        # Camera forward is typically +Z axis, so extract yaw from Z-column
        R = self._dr_rotation_matrix
        hloc_yaw_rad = np.arctan2(R[0, 2], R[2, 2])  # Yaw of forward (+Z) direction
        hloc_yaw_deg = np.degrees(hloc_yaw_rad)

        # Extract telemetry heading
        telem_heading_deg = self._latest_telemetry.get('heading_deg', 0.0)

        # Compute heading offset (handle wrap-around)
        heading_offset = hloc_yaw_deg - telem_heading_deg
        # Normalize to [-180, 180]
        self._heading_offset_deg = ((heading_offset + 180.0) % 360.0) - 180.0

        # Extract altitude offset (Y is altitude in HLOC)
        # Negate position_z_m because NED Z-down (positive=down) vs HLOC Y-up (positive=up)
        telem_altitude = -self._latest_telemetry.get('position_z_m', 0.0)
        hloc_y = self._dr_position[1]  # Y is altitude
        self._altitude_offset_m = hloc_y - telem_altitude

        self._calibrated = True

        logging.info(f"Calibration complete:")
        logging.info(f"  Heading offset: {self._heading_offset_deg:.2f}°")
        logging.info(f"  Altitude offset: {self._altitude_offset_m:.2f}m")
        logging.info(f"  HLOC yaw: {hloc_yaw_deg:.2f}°, Telem heading: {telem_heading_deg:.2f}°")

    def _update_dead_reckoning(self, prev_telem, curr_telem, prev_time):
        """Integrate velocities to update dead reckoning position"""

        # Calculate time delta
        current_time = datetime.now()
        dt = (current_time - prev_time).total_seconds()

        # Sanity check: skip if dt is too large (missing data) or too small (duplicate)
        if dt <= 0.0 or dt > 1.0:
            logging.warning(f"Abnormal dt={dt:.3f}s, skipping dead reckoning update")
            return

        # Extract NED velocities (m/s)
        v_ned = np.array([
            curr_telem.get('velocity_x_ms', 0.0),  # North
            curr_telem.get('velocity_y_ms', 0.0),  # East
            curr_telem.get('velocity_z_ms', 0.0),  # Down
        ])

        # Transform velocities: NED → HLOC world frame
        # HLOC: Y-up, horizontal plane is X-Z
        # If forward is +Z in HLOC, then: North→Z, East→X (or vice versa)
        theta = np.radians(self._heading_offset_deg)
        v_hloc_z = v_ned[0] * np.cos(theta) - v_ned[1] * np.sin(theta)  # North→Z
        v_hloc_x = v_ned[0] * np.sin(theta) + v_ned[1] * np.cos(theta)  # East→X
        v_hloc_y = -v_ned[2]  # Vertical (NED Z-down → HLOC Y-up)

        # Integrate horizontal position (X, Z from velocities)
        self._dr_position[0] += v_hloc_x * dt
        self._dr_position[2] += v_hloc_z * dt

        # Use altitude directly for Y (more accurate than integrating velocity_y)
        # Negate position_z_m because NED Z-down (positive=down) vs HLOC Y-up (positive=up)
        telem_altitude = -curr_telem.get('position_z_m', 0.0)
        self._dr_position[1] = telem_altitude + self._altitude_offset_m  # Y is altitude

        # Update orientation (yaw only from telemetry heading)
        telem_heading = curr_telem.get('heading_deg', 0.0)
        hloc_yaw = np.radians(telem_heading + self._heading_offset_deg)

        # Construct rotation matrix (Y-up convention, yaw rotation around Y)
        cos_yaw = np.cos(hloc_yaw)
        sin_yaw = np.sin(hloc_yaw)
        self._dr_rotation_matrix = np.array([
            [ cos_yaw, 0.0, sin_yaw],
            [ 0.0,     1.0, 0.0    ],
            [-sin_yaw, 0.0, cos_yaw]
        ])

    def set_error(self, error_msg):
        """Set error message (keeps last good pose)"""
        with self._lock:
            self._error_message = error_msg

    def get_latest(self):
        """Get latest position estimate for visualization"""
        with self._lock:
            if self._pose_source == "waiting":
                return None

            # Return current position/rotation from dead reckoning
            if self._dr_position is None:
                return None

            return {
                'position': self._dr_position.copy(),
                'rotation': self._dr_rotation_matrix.copy(),
                'cam_from_world': self._latest_hloc_pose,  # Keep for camera params
                'camera': self._latest_hloc_camera,
                'inliers': self._latest_hloc_inliers,
                'total_matches': self._latest_hloc_matches,
                'frame_num': self._latest_hloc_frame,
                'timestamp': self._latest_hloc_timestamp,
                'error': self._error_message,
                'source': self._pose_source,
                'calibrated': self._calibrated,
            }


# Cache for sparse model bounds (computed once per model)
_bounds_cache = {}

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

    # Get cached bounds or compute them (expensive operation, only do once)
    model_id = id(sparse_model)
    if model_id not in _bounds_cache:
        all_points = np.array([p3D.xyz for p3D in sparse_model.points3D.values()])
        data_min = np.percentile(all_points, 0.1, axis=0)
        data_max = np.percentile(all_points, 99.9, axis=0)
        data_center = (data_min + data_max) / 2.0
        data_extent = data_max - data_min
        max_extent = np.max(data_extent)
        scale_factor = 2.0 / max_extent
        _bounds_cache[model_id] = (data_center, scale_factor)

    data_center, scale_factor = _bounds_cache[model_id]

    # Convert to Plotly domain coordinates
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
    app = Dash(__name__, suppress_callback_exceptions=True)

    # Add Flask routes for Gaussian Splat viewer
    @app.server.route('/api/drone_position')
    def get_drone_position():
        """API endpoint for Gaussian Splat viewer to get drone position"""
        latest = localization_state.get_latest()
        if latest is None:
            response = jsonify({'error': 'No position data'})
            response.status_code = 404
        else:
            response = jsonify({
                'position': latest['position'].tolist(),
                'rotation': latest['rotation'].flatten().tolist(),  # 3x3 matrix as flat array
                'source': latest['source'],
                'calibrated': latest['calibrated'],
                'inliers': latest['inliers'],
                'total_matches': latest['total_matches'],
                'frame_num': latest['frame_num'],
            })
        
        return response

    @app.server.route('/splat/<path:filename>')
    def serve_splat(filename):
        """Serve Gaussian Splat PLY file"""
        splat_dir = Path('final_inputs/v1/splats')
        return send_from_directory(splat_dir, filename)

    @app.server.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files (splat viewer HTML/JS)"""
        return send_from_directory('static', filename)

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

    # Store number of base traces (for Patch-based updates)
    num_base_traces = len(base_fig.data)
    logger.info(f"Base figure has {num_base_traces} traces")

    # Add placeholder drone traces that will be updated via Patch
    # Trace 1: Camera frustum (lines forming pyramid shape)
    # Frustum has 5 points: camera center + 4 corners of image plane
    # We draw lines: center->corners and around the rectangle
    base_fig.add_trace(go.Scatter3d(
        x=[0]*16, y=[0]*16, z=[0]*16,
        mode='lines',
        line=dict(width=3, color='green'),
        name='Drone Camera',
        showlegend=True,
        connectgaps=False  # None values create gaps between line segments
    ))
    # Trace 2: Frustum fill (mesh for the front face)
    base_fig.add_trace(go.Mesh3d(
        x=[0]*4, y=[0]*4, z=[0]*4,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='green',
        opacity=0.3,
        name='Frustum Fill',
        showlegend=False
    ))

    # Track if first update (need full figure) or subsequent (use Patch)
    first_update = [True]  # Use list for mutability in closure

    app.layout = html.Div([
        html.H1("Real-Time Drone Localization", style={'textAlign': 'center'}),

        # Status display
        html.Div(id='status-text', style={'padding': '20px', 'fontSize': '14px'}),

        # Controls row
        html.Div([
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
            ], style={'display': 'inline-block', 'marginRight': '40px'}),

            # Visualization mode toggle
            html.Div([
                html.Label("Visualization:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.RadioItems(
                    id='viz-mode-toggle',
                    options=[
                        {'label': ' Sparse Point Cloud', 'value': 'sparse'},
                        {'label': ' Gaussian Splat', 'value': 'splat'}
                    ],
                    value='sparse',  # Default to sparse
                    labelStyle={'display': 'inline-block', 'marginRight': '20px'},
                    style={'display': 'inline-block'}
                )
            ], style={'display': 'inline-block'}),
        ], style={'padding': '10px 20px', 'backgroundColor': '#f0f0f0', 'marginBottom': '10px'}),

        # Store for camera state in manual mode
        dcc.Store(id='camera-store', data=None),

        # Store for tracking last camera interaction time
        dcc.Store(id='camera-interaction-time', data=0),

        # 3D visualization container (shows either Plotly or iframe)
        html.Div(id='viz-container', children=[
            # Sparse point cloud (Plotly)
            dcc.Graph(
                id='3d-plot',
                figure=base_fig,
                style={'height': '80vh'}
            ),
            # Gaussian Splat viewer (iframe - hidden by default)
            html.Iframe(
                id='splat-viewer',
                src='/static/splat_viewer.html',
                style={'width': '100%', 'height': '80vh', 'border': 'none', 'display': 'none'}
            ),
        ]),

        # Auto-refresh interval (every 1 second)
        dcc.Interval(
            id='interval-component',
            interval=DASH_REFRESH_INTERVAL,  # milliseconds
            n_intervals=0
        ),

    ])

    # Clientside callback to capture camera state when user interacts with the plot
    app.clientside_callback(
        """
        function(relayoutData, currentStoredCamera) {
            if (!relayoutData) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            // Check if camera data is in relayoutData
            let cameraData = null;
            if ('scene.camera' in relayoutData) {
                cameraData = relayoutData['scene.camera'];
            }
            // Also check for individual camera properties
            else if ('scene.camera.eye.x' in relayoutData) {
                cameraData = {
                    eye: {
                        x: relayoutData['scene.camera.eye.x'],
                        y: relayoutData['scene.camera.eye.y'],
                        z: relayoutData['scene.camera.eye.z']
                    },
                    center: {
                        x: relayoutData['scene.camera.center.x'] || 0,
                        y: relayoutData['scene.camera.center.y'] || 0,
                        z: relayoutData['scene.camera.center.z'] || 0
                    },
                    up: {
                        x: relayoutData['scene.camera.up.x'] || 0,
                        y: relayoutData['scene.camera.up.y'] || 1,
                        z: relayoutData['scene.camera.up.z'] || 0
                    }
                };
            }
            if (cameraData) {
                // Check if camera actually changed from what we have stored
                // (to avoid triggering debounce from our own programmatic updates)
                let cameraChanged = true;
                if (currentStoredCamera && currentStoredCamera.eye) {
                    const threshold = 0.0001;
                    const eyeMatch = Math.abs(cameraData.eye.x - currentStoredCamera.eye.x) < threshold &&
                                     Math.abs(cameraData.eye.y - currentStoredCamera.eye.y) < threshold &&
                                     Math.abs(cameraData.eye.z - currentStoredCamera.eye.z) < threshold;
                    if (eyeMatch) {
                        cameraChanged = false;
                    }
                }
                if (cameraChanged) {
                    return [cameraData, Date.now()];
                } else {
                    // Camera didn't change (programmatic update), just store it without updating timestamp
                    return [cameraData, window.dash_clientside.no_update];
                }
            }
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        """,
        [Output('camera-store', 'data'),
         Output('camera-interaction-time', 'data')],
        [Input('3d-plot', 'relayoutData')],
        [State('camera-store', 'data')],
        prevent_initial_call=True
    )

    # Callback to toggle visualization visibility
    @app.callback(
        [Output('3d-plot', 'style'),
         Output('splat-viewer', 'style')],
        [Input('viz-mode-toggle', 'value')]
    )
    def toggle_visualization(viz_mode):
        """Toggle between sparse point cloud and Gaussian splat visualization"""
        if viz_mode == 'sparse':
            return {'height': '80vh'}, {'display': 'none'}
        else:
            return {'display': 'none'}, {'width': '100%', 'height': '80vh', 'border': 'none'}

    # Clientside callback to send camera mode to iframe
    app.clientside_callback(
        """
        function(cameraMode) {
            const iframe = document.getElementById('splat-viewer');
            if (iframe && iframe.contentWindow) {
                iframe.contentWindow.postMessage({type: 'setCameraMode', mode: cameraMode}, '*');
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('splat-viewer', 'id'),  # Dummy output (no actual change)
        [Input('camera-mode-toggle', 'value')],
        prevent_initial_call=True
    )

    @app.callback(
        [Output('3d-plot', 'figure'),
         Output('status-text', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('camera-mode-toggle', 'value')],
        [State('camera-store', 'data'),
         State('camera-interaction-time', 'data')]
    )
    def update_visualization(n, camera_mode, stored_camera, last_interaction_time):
        """Update 3D visualization with latest localization data using Patch for speed"""
        # In manual mode, skip updates for 1 second after camera interaction
        if camera_mode == 'manual' and last_interaction_time:
            elapsed_ms = time.time() * 1000 - last_interaction_time
            if elapsed_ms < 2000:  # 2 second debounce
                return no_update, no_update

        # Get latest localization data from shared state
        latest = localization_state.get_latest()
        logger.debug(f"Dash callback: latest={latest is not None}")

        if latest is None:
            # No localization yet - show base figure with waiting message
            logger.debug("  No localization data yet")
            if first_update[0]:
                return base_fig, html.Div("Waiting for first localization...", style={'color': 'gray', 'fontSize': '16px'})
            else:
                return no_update, html.Div("Waiting for first localization...", style={'color': 'gray', 'fontSize': '16px'})

        try:
            logger.debug(f"  Rendering visualization for frame {latest['frame_num']}")

            # Get drone position and orientation from dead reckoning
            drone_pos = latest['position']
            drone_rot = latest['rotation']

            # Choose color based on source: green for HLOC, orange for telemetry
            source = latest['source']
            frustum_color = 'green' if source == 'hloc' else 'orange'

            # Calculate camera frustum corners
            # Camera axes from rotation matrix (columns)
            right = drone_rot[:, 0]    # X-axis (right)
            up = -drone_rot[:, 1]      # Y-axis (up, negated because camera Y is down)
            forward = drone_rot[:, 2]  # Z-axis (forward)

            # Frustum parameters (smaller for cleaner visualization)
            frustum_depth = 0.25  # How far the frustum extends (meters)
            aspect = 16/9         # Aspect ratio
            fov_scale = 0.2       # Controls frustum width

            # Calculate the 4 corners of the frustum front face
            half_w = frustum_depth * fov_scale * aspect
            half_h = frustum_depth * fov_scale
            center_front = drone_pos + forward * frustum_depth

            # Corners: top-left, top-right, bottom-right, bottom-left
            c_tl = center_front + up * half_h - right * half_w
            c_tr = center_front + up * half_h + right * half_w
            c_br = center_front - up * half_h + right * half_w
            c_bl = center_front - up * half_h - right * half_w

            # Build frustum lines: center to each corner, then around the rectangle
            # Format: [center->tl, None, center->tr, None, center->br, None, center->bl, None, tl->tr->br->bl->tl]
            frustum_x = [drone_pos[0], c_tl[0], None, drone_pos[0], c_tr[0], None,
                        drone_pos[0], c_br[0], None, drone_pos[0], c_bl[0], None,
                        c_tl[0], c_tr[0], c_br[0], c_bl[0], c_tl[0]]
            frustum_y = [drone_pos[1], c_tl[1], None, drone_pos[1], c_tr[1], None,
                        drone_pos[1], c_br[1], None, drone_pos[1], c_bl[1], None,
                        c_tl[1], c_tr[1], c_br[1], c_bl[1], c_tl[1]]
            frustum_z = [drone_pos[2], c_tl[2], None, drone_pos[2], c_tr[2], None,
                        drone_pos[2], c_br[2], None, drone_pos[2], c_bl[2], None,
                        c_tl[2], c_tr[2], c_br[2], c_bl[2], c_tl[2]]

            # Drone trace indices (after base traces)
            frustum_trace_idx = num_base_traces      # Frustum lines
            mesh_trace_idx = num_base_traces + 1     # Frustum fill mesh

            # Use Patch for fast partial updates (no deepcopy!)
            patched_fig = Patch()

            # Update frustum lines
            patched_fig['data'][frustum_trace_idx]['x'] = frustum_x
            patched_fig['data'][frustum_trace_idx]['y'] = frustum_y
            patched_fig['data'][frustum_trace_idx]['z'] = frustum_z
            patched_fig['data'][frustum_trace_idx]['line']['color'] = frustum_color
            patched_fig['data'][frustum_trace_idx]['name'] = f'Drone ({source.upper()})'

            # Update frustum fill mesh (4 corners of front face)
            patched_fig['data'][mesh_trace_idx]['x'] = [c_tl[0], c_tr[0], c_br[0], c_bl[0]]
            patched_fig['data'][mesh_trace_idx]['y'] = [c_tl[1], c_tr[1], c_br[1], c_bl[1]]
            patched_fig['data'][mesh_trace_idx]['z'] = [c_tl[2], c_tr[2], c_br[2], c_bl[2]]
            patched_fig['data'][mesh_trace_idx]['color'] = frustum_color

            # Calculate and apply camera based on mode
            if camera_mode == 'auto':
                logger.debug("  Calculating third-person camera position...")
                camera_config = calculate_third_person_camera(
                    drone_pos,
                    drone_rot,
                    localization_state.sparse_model,
                    offset_back=CAMERA_OFFSET_BACK,
                    offset_up=CAMERA_OFFSET_UP,
                    look_ahead=CAMERA_LOOK_AHEAD
                )
                camera_config['projection'] = {'type': 'perspective'}
                patched_fig['layout']['scene']['camera'] = camera_config
                patched_fig['layout']['uirevision'] = n
            else:
                # Manual mode: apply stored camera to preserve user's view
                if stored_camera:
                    patched_fig['layout']['scene']['camera'] = stored_camera
                patched_fig['layout']['uirevision'] = 'manual'

            # First update needs full figure, subsequent use Patch
            if first_update[0]:
                first_update[0] = False
                # Apply updates to base_fig for first render
                base_fig.data[frustum_trace_idx].x = frustum_x
                base_fig.data[frustum_trace_idx].y = frustum_y
                base_fig.data[frustum_trace_idx].z = frustum_z
                base_fig.data[frustum_trace_idx].line.color = frustum_color
                base_fig.data[mesh_trace_idx].x = [c_tl[0], c_tr[0], c_br[0], c_bl[0]]
                base_fig.data[mesh_trace_idx].y = [c_tl[1], c_tr[1], c_br[1], c_bl[1]]
                base_fig.data[mesh_trace_idx].z = [c_tl[2], c_tr[2], c_br[2], c_bl[2]]
                base_fig.data[mesh_trace_idx].color = frustum_color
                fig_output = base_fig
            else:
                fig_output = patched_fig

            # Build status text
            logger.debug("  Building status text...")
            pos = drone_pos
            rot = Rotation.from_matrix(drone_rot)
            yaw, pitch, roll = rot.as_euler('zyx', degrees=True)

            # Get source and calibration info
            calibrated = latest['calibrated']
            source_color = 'green' if source == 'hloc' else 'orange'

            status_children = [
                html.H3("Drone Position (meters)"),
                html.P(f"X: {pos[0]:.3f} | Y: {pos[1]:.3f} | Z: {pos[2]:.3f}"),
                html.H3("Orientation (degrees)"),
                html.P(f"Yaw: {yaw:.1f}° | Pitch: {pitch:.1f}° | Roll: {roll:.1f}°"),
                html.H3("Localization Info"),
                html.P([
                    f"Frame: {latest['frame_num']} | ",
                    html.Span(
                        f"Source: {source.upper()}",
                        style={'fontWeight': 'bold', 'color': source_color}
                    ),
                    f" | HLOC Inliers: {latest['inliers']}/{latest['total_matches']} | ",
                    f"Calibrated: {'✓' if calibrated else '✗'} | ",
                    f"Time: {latest['timestamp'].strftime('%H:%M:%S')}"
                ])
            ]

            # Add error message if present
            if latest['error']:
                status_children.append(
                    html.Div(
                        f"⚠ Warning: {latest['error']}",
                        style={'color': 'red', 'fontWeight': 'bold', 'marginTop': '10px'}
                    )
                )

            logger.debug("  Visualization rendered successfully")
            return fig_output, html.Div(status_children)
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
    # Silence Flask/Werkzeug HTTP request logs
    import logging as werkzeug_logging
    werkzeug_log = werkzeug_logging.getLogger('werkzeug')
    werkzeug_log.setLevel(werkzeug_logging.WARNING)

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

        # Initialize shared state for visualization (hybrid HLOC + telemetry)
        self.localization_state = HybridLocalizationState(self.sparse_model)

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

            @room.on("data_received")
            def on_data_received(data: rtc.DataPacket):
                """Receive and process telemetry data packets"""
                try:
                    telemetry = json.loads(data.data.decode('utf-8'))
                    self.localization_state.update_telemetry(telemetry)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse telemetry JSON: {e}")
                except Exception as e:
                    logger.error(f"Failed to process telemetry: {e}")

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
                # Success - update shared state with HLOC fix
                self.localization_state.update_hloc_pose({
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
