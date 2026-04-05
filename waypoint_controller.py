"""
Waypoint Mission Controller

Cloud-side controller for autonomous drone waypoint missions.

Responsibilities:
- Load waypoints from YAML
- Pre-flight collision validation against occupancy grid
- WebSocket server for Android communication
- Send hloc_init to Android after first HLOC fix
- Iterate waypoints: send → wait for "arrived" → send next
- Send hloc_correction each time HLOC produces a new fix mid-mission
- Handle "collision" from Android → replan with A* → send new path
"""

import asyncio
import json
import logging
from typing import Optional

import numpy as np
import yaml
import websockets
from websockets.server import WebSocketServerProtocol

from occupancy_grid import OccupancyGrid

logger = logging.getLogger(__name__)

WEBSOCKET_PORT = 8765


class WaypointController:
    def __init__(self, waypoints_path: str, grid: OccupancyGrid):
        """
        Args:
            waypoints_path: Path to YAML file with list of [x, y, z] waypoints
                            in HLOC world frame.
            grid: Loaded OccupancyGrid instance.

        Raises:
            SystemExit: if pre-flight validation finds collisions.
        """
        self._grid = grid
        self._waypoints = self._load_waypoints(waypoints_path)
        self._localization_state = None

        # WebSocket state
        self._connected_client: Optional[WebSocketServerProtocol] = None
        self._arrival_queue: asyncio.Queue = asyncio.Queue()
        self._start_mission_event: asyncio.Event = asyncio.Event()

        # HLOC correction tracking (detect new fixes by frame_num change)
        self._last_sent_frame_num: Optional[int] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_waypoints(self, path: str) -> list:
        with open(path) as f:
            data = yaml.safe_load(f)
        self._arrival_threshold: float = float(data.get('arrival_threshold', 0.15))
        waypoints = [np.array(wp, dtype=float) for wp in data['waypoints']]
        logger.info(f"Loaded {len(waypoints)} waypoints from {path}, "
                    f"arrival_threshold={self._arrival_threshold}m")
        return waypoints

    def validate_mission(self) -> list[str]:
        """
        Run pre-flight collision check.

        Returns list of collision descriptions (empty = all clear).
        """
        return self._grid.validate_mission(self._waypoints)

    def set_localization_state(self, localization_state):
        self._localization_state = localization_state

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self):
        """
        Start the controller. Blocks until mission completes or is aborted.

        Sequence:
        1. Start WebSocket server immediately
        2. Wait for Android to connect (Android starts LiveKit stream on connect)
        3. Wait for first HLOC fix (frames are now flowing via LiveKit)
        4. Run mission_loop() and hloc_correction_loop() concurrently
        """
        async with websockets.serve(self._handle_android, "0.0.0.0", WEBSOCKET_PORT):
            logger.info(f"WaypointController: WebSocket server listening on port {WEBSOCKET_PORT}")
            logger.info("WaypointController: waiting for Android to connect...")
            await self._wait_for_client_connection()
            logger.info("WaypointController: Android connected — waiting for first HLOC fix...")
            await self._wait_for_hloc_fix()
            logger.info("WaypointController: first HLOC fix received — starting mission")

            await asyncio.gather(
                self._mission_loop(),
                self._hloc_correction_loop(),
            )

    # ------------------------------------------------------------------
    # Mission loop
    # ------------------------------------------------------------------

    async def _mission_loop(self):
        # Send hloc_init so Android can initialize position estimator
        await self._send_hloc_init()

        # Wait for user to tap "Start Mission" on Android
        logger.info("Waiting for start_mission from Android...")
        await self._start_mission_event.wait()
        logger.info("start_mission received — sending waypoints")

        remaining_waypoints = list(self._waypoints)
        waypoint_index = 0

        while remaining_waypoints:
            target = remaining_waypoints[0]
            logger.info(f"Waypoint {waypoint_index}: sending {target.tolist()}")
            await self._send_waypoint(target)

            # Wait for "arrived" or "collision"
            msg = await self._arrival_queue.get()
            msg_type = msg.get("type")

            if msg_type == "arrived":
                logger.info(f"Waypoint {waypoint_index}: arrived")
                remaining_waypoints.pop(0)
                waypoint_index += 1

            elif msg_type == "collision":
                logger.warning(f"Waypoint {waypoint_index}: collision reported by Android")
                current_pos = self._localization_state.get_latest()
                if current_pos is None:
                    logger.error("Cannot replan — no localization data")
                    break

                drone_pos = current_pos['position']
                new_path = self._grid.astar(drone_pos, target)

                if new_path is None:
                    logger.error(f"A* replanning failed — no path to waypoint {waypoint_index}")
                    break

                logger.info(f"Replanned path: {len(new_path)} intermediate waypoints")
                # Replace head of remaining with replanned intermediate points
                # (the original target stays as final destination)
                remaining_waypoints = new_path + remaining_waypoints[1:]

            else:
                logger.warning(f"Unknown message from Android: {msg}")

        await self._broadcast({"type": "mission_complete"})
        logger.info("Mission complete")

    # ------------------------------------------------------------------
    # HLOC correction loop
    # ------------------------------------------------------------------

    async def _hloc_correction_loop(self):
        """
        Poll localization state every 0.5s.
        When a new HLOC fix arrives (frame_num changes, source == "hloc"),
        send hloc_correction to Android.
        """
        while True:
            await asyncio.sleep(0.5)

            if self._localization_state is None:
                continue

            latest = self._localization_state.get_latest()
            if latest is None:
                continue

            # Only send corrections when a new HLOC frame has been processed.
            # (source flips to "telemetry" immediately after each fix, so we
            # detect new fixes by frame_num change instead of source check.)
            frame_num = latest.get('frame_num')
            if frame_num is None:
                continue

            # Detect a new HLOC fix by frame_num change
            if frame_num == self._last_sent_frame_num:
                continue

            # Skip if this is the same fix we used for hloc_init
            if self._last_sent_frame_num is None:
                # hloc_init already covers this fix
                self._last_sent_frame_num = frame_num
                continue

            self._last_sent_frame_num = frame_num
            await self._send_hloc_correction(latest)

    # ------------------------------------------------------------------
    # WebSocket handlers
    # ------------------------------------------------------------------

    async def _handle_android(self, websocket: WebSocketServerProtocol):
        logger.info(f"Android connected from {websocket.remote_address}")
        self._connected_client = websocket
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                    msg_type = msg.get("type")
                    if msg_type == "start_mission":
                        self._start_mission_event.set()
                    elif msg_type in ("arrived", "collision"):
                        await self._arrival_queue.put(msg)
                    else:
                        logger.warning(f"Unrecognized message from Android: {msg}")
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message from Android: {raw!r}")
        finally:
            logger.info("Android disconnected")
            if self._connected_client is websocket:
                self._connected_client = None

    async def _wait_for_client_connection(self):
        while self._connected_client is None:
            await asyncio.sleep(0.1)

    async def _wait_for_hloc_fix(self):
        # Wait for calibrated == True.
        # Checking source == 'hloc' is unreliable because telemetry arrives
        # immediately after each HLOC fix and overwrites source to "telemetry".
        while True:
            if self._localization_state is not None:
                latest = self._localization_state.get_latest()
                if latest is not None and latest.get('calibrated'):
                    return
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # Message senders
    # ------------------------------------------------------------------

    async def _broadcast(self, msg: dict):
        if self._connected_client is None:
            logger.warning(f"Cannot send {msg['type']} — no client connected")
            return
        try:
            await self._connected_client.send(json.dumps(msg))
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"Send failed — connection closed")

    async def _send_hloc_init(self):
        latest = self._localization_state.get_latest()
        if latest is None:
            logger.error("send_hloc_init: no localization data available")
            return

        msg = {
            "type": "hloc_init",
            "position": latest['position'].tolist(),
            "heading_offset_deg": latest['heading_offset_deg'],
            "altitude_offset_m": latest['altitude_offset_m'],
            "arrival_threshold_m": self._arrival_threshold,
        }
        self._last_sent_frame_num = latest.get('frame_num')
        await self._broadcast(msg)
        logger.info(f"Sent hloc_init: pos={latest['position'].tolist()}, "
                    f"heading_offset={latest['heading_offset_deg']:.2f}°")

    async def _send_hloc_correction(self, latest: dict):
        msg = {
            "type": "hloc_correction",
            "position": latest['position'].tolist(),
            "heading_offset_deg": latest['heading_offset_deg'],
            "altitude_offset_m": latest['altitude_offset_m'],
        }
        await self._broadcast(msg)
        logger.info(f"Sent hloc_correction: pos={latest['position'].tolist()}, "
                    f"heading_offset={latest['heading_offset_deg']:.2f}°")

    async def _send_waypoint(self, point: np.ndarray):
        msg = {
            "type": "waypoint",
            "position": point.tolist(),
        }
        await self._broadcast(msg)
        logger.info(f"Sent waypoint: {point.tolist()}")
