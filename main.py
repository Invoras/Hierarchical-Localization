"""
Waypoint Mission Entry Point

Wires together:
- OccupancyGrid (static + dynamic obstacle map)
- WaypointController (mission sequencing, WebSocket server)
- FrameCapture / HybridLocalizationState (HLOC + dead reckoning)

Usage:
    python main.py

The localization Dash visualization is still available at http://127.0.0.1:8050
"""

import asyncio
import logging
import os
import sys

# Allow importing from Hierarchical-Localization directory (where this file lives)
sys.path.insert(0, os.path.dirname(__file__))

from live_localization import FrameCapture
from occupancy_grid import OccupancyGrid
from waypoint_controller import WaypointController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Paths ---
GRID_PATH = os.path.join(os.path.dirname(__file__), '../Occupancygrid_and_Waypoint/maps/occupancy_grid.npy')
META_PATH = os.path.join(os.path.dirname(__file__), '../Occupancygrid_and_Waypoint/maps/occupancy_grid_meta.npy')
WAYPOINTS_PATH = os.path.join(os.path.dirname(__file__), 'waypoints.yaml')

# Localization model paths (same as used by live_localization.py standalone)
SPARSE_MODEL_PATH = "final_inputs/v1/sparse/0"
REFERENCE_FEATURES_PATH = "outputs/sfm_1s/feats-superpoint-n4096-r1024.h5"
REFERENCE_IMAGES_PATH = "final_inputs/v1/images"
OUTPUT_FOLDER = "outputs/localization_temp"


async def main():
    # 1. Load occupancy grid
    grid = OccupancyGrid.load(GRID_PATH, META_PATH)

    # 2. Load waypoints and run pre-flight validation
    controller = WaypointController(WAYPOINTS_PATH, grid)
    issues = controller.validate_mission()
    if issues:
        logger.error("Mission has collisions — fix waypoints before flying:")
        for issue in issues:
            logger.error(f"  {issue}")
        logger.info("Opening visualization — red = obstacle/blocked, green = clear")
        grid.visualize_mission(controller._waypoints)
        return

    logger.info("Mission clear — no pre-flight collisions detected")

    # 3. Start localization
    capture = FrameCapture(
        output_folder=OUTPUT_FOLDER,
        sparse_model_path=SPARSE_MODEL_PATH,
        reference_features_path=REFERENCE_FEATURES_PATH,
        reference_images_path=REFERENCE_IMAGES_PATH,
    )

    # 4. Give controller access to live localization state
    controller.set_localization_state(capture.localization_state)

    # 5. Run everything concurrently
    await asyncio.gather(
        capture.run(),       # HLOC localization loop + Dash visualization
        controller.run(),    # Waypoint mission loop + WebSocket server
    )


if __name__ == "__main__":
    asyncio.run(main())
