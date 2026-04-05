"""
Occupancy Grid for indoor warehouse drone navigation.

Loads the static grid built by build_grid.py and supports:
- Segment collision checking (ray marching)
- Pre-flight mission validation
- Dynamic obstacle marking from perception data
- 3D A* path replanning
"""

import asyncio
import heapq
import logging
from typing import Optional

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


class OccupancyGrid:
    def __init__(self, grid: np.ndarray, origin: np.ndarray, voxel_size: float):
        """
        Args:
            grid: (nx, ny, nz) bool array — True = occupied
            origin: [x, y, z] HLOC world coords of voxel [0, 0, 0]
            voxel_size: meters per voxel (0.05)
        """
        self.grid = grid
        self.origin = origin.astype(float)
        self.voxel_size = float(voxel_size)
        self._dynamic_grid = np.zeros_like(grid, dtype=bool)
        self._dynamic_lock = asyncio.Lock()

    @classmethod
    def load(cls, grid_path: str, meta_path: str) -> "OccupancyGrid":
        grid = np.load(grid_path)
        meta = np.load(meta_path)
        origin = meta[:3]
        voxel_size = float(meta[3])
        logger.info(f"Loaded occupancy grid: shape={grid.shape}, voxel_size={voxel_size}m, "
                    f"occupied={grid.sum()} voxels")
        return cls(grid, origin, voxel_size)

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def world_to_voxel(self, point: np.ndarray) -> tuple[int, int, int]:
        """Convert HLOC world-frame point to (ix, iy, iz) voxel indices."""
        idx = ((np.asarray(point) - self.origin) / self.voxel_size).astype(int)
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def voxel_to_world(self, voxel: tuple) -> np.ndarray:
        """Convert voxel indices to HLOC world-frame center point."""
        return np.array(voxel, dtype=float) * self.voxel_size + self.origin

    def _in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        nx, ny, nz = self.grid.shape
        return 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz

    def is_occupied(self, voxel: tuple) -> bool:
        """Check if voxel is occupied (static or dynamic layer).
        Out-of-bounds voxels are treated as free space — the drone may fly
        slightly outside the scanned area (e.g. above the highest scanned point).
        """
        ix, iy, iz = voxel
        if not self._in_bounds(ix, iy, iz):
            return False
        return bool(self.grid[ix, iy, iz] or self._dynamic_grid[ix, iy, iz])

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def check_segment(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """
        Check if the straight-line segment from p1 to p2 is collision-free.

        Returns True if the path is clear, False if it hits an obstacle.
        Samples at voxel_size/2 intervals.
        """
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        dist = float(np.linalg.norm(p2 - p1))
        if dist == 0.0:
            return not self.is_occupied(self.world_to_voxel(p1))

        step = self.voxel_size / 2.0
        n_steps = int(np.ceil(dist / step)) + 1
        for i in range(n_steps + 1):
            t = min(i / n_steps, 1.0)
            pt = p1 + t * (p2 - p1)
            voxel = self.world_to_voxel(pt)
            if self.is_occupied(voxel):
                return False
        return True

    def validate_mission(self, waypoints: list) -> list[str]:
        """
        Pre-flight validation of a list of HLOC world-frame waypoints.

        Checks:
        - Each waypoint position is not in an occupied voxel
        - Each segment between consecutive waypoints is collision-free

        Returns list of human-readable collision descriptions (empty = all clear).
        """
        issues = []
        waypoints = [np.asarray(wp, dtype=float) for wp in waypoints]

        for i, wp in enumerate(waypoints):
            voxel = self.world_to_voxel(wp)
            if self.is_occupied(voxel):
                issues.append(f"Waypoint {i} at {wp.tolist()} is inside an obstacle")

        for i in range(len(waypoints) - 1):
            if not self.check_segment(waypoints[i], waypoints[i + 1]):
                issues.append(
                    f"Path from waypoint {i} to {i + 1} "
                    f"({waypoints[i].tolist()} → {waypoints[i + 1].tolist()}) is blocked"
                )

        return issues

    # ------------------------------------------------------------------
    # A* path replanning
    # ------------------------------------------------------------------

    def astar(self, start: np.ndarray, goal: np.ndarray) -> Optional[list]:
        """
        3D A* on the combined static + dynamic grid.
        26-connectivity (6 face + 12 edge + 8 corner neighbors).

        Returns list of HLOC world-frame waypoints from start to goal,
        or None if no path exists.
        """
        start_v = self.world_to_voxel(np.asarray(start, dtype=float))
        goal_v = self.world_to_voxel(np.asarray(goal, dtype=float))

        if self.is_occupied(goal_v):
            logger.warning(f"A*: goal voxel {goal_v} is occupied, cannot replan")
            return None

        def h(v):
            # Euclidean heuristic in voxel space
            return float(np.linalg.norm(np.array(v) - np.array(goal_v)))

        # (f, g, voxel)
        open_heap: list = []
        heapq.heappush(open_heap, (h(start_v), 0.0, start_v))
        came_from: dict = {}
        g_score: dict = {start_v: 0.0}

        # 26-connectivity offsets
        neighbors = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        # diagonal step costs
        diag_cost = {
            1: 1.0,
            2: 1.414,
            3: 1.732,
        }

        while open_heap:
            f, g, current = heapq.heappop(open_heap)

            if current == goal_v:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(self.voxel_to_world(node))
                    node = came_from[node]
                path.reverse()
                path.append(self.voxel_to_world(goal_v))
                logger.info(f"A*: found path with {len(path)} waypoints")
                return path

            for dx, dy, dz in neighbors:
                nx_ = current[0] + dx
                ny_ = current[1] + dy
                nz_ = current[2] + dz
                neighbor = (nx_, ny_, nz_)

                if not self._in_bounds(nx_, ny_, nz_):
                    continue
                if self.is_occupied(neighbor):
                    continue

                step_cost = diag_cost[abs(dx) + abs(dy) + abs(dz)]
                tentative_g = g + step_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_heap, (tentative_g + h(neighbor), tentative_g, neighbor))

        logger.warning("A*: no path found")
        return None

    # ------------------------------------------------------------------
    # Dynamic obstacle updates
    # ------------------------------------------------------------------

    async def mark_dynamic_obstacle(self, center: np.ndarray, radius_m: float):
        """
        Mark a sphere of given radius (meters) as occupied in the dynamic layer.
        Called when Android reports a collision at estimated position.
        """
        center = np.asarray(center, dtype=float)
        radius_voxels = int(np.ceil(radius_m / self.voxel_size))
        cx, cy, cz = self.world_to_voxel(center)
        nx, ny, nz = self.grid.shape

        async with self._dynamic_lock:
            for dx in range(-radius_voxels, radius_voxels + 1):
                for dy in range(-radius_voxels, radius_voxels + 1):
                    for dz in range(-radius_voxels, radius_voxels + 1):
                        if dx * dx + dy * dy + dz * dz <= radius_voxels * radius_voxels:
                            ix, iy, iz = cx + dx, cy + dy, cz + dz
                            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                                self._dynamic_grid[ix, iy, iz] = True

        logger.info(f"Marked dynamic obstacle at {center.tolist()} radius={radius_m}m")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_mission(self, waypoints: list):
        """
        Open an Open3D window showing:
        - Occupancy grid voxels as a point cloud (Open3D default coloring)
        - Waypoints as spheres: green = free space, red = inside obstacle
        - Paths between consecutive waypoints: green = clear, red = blocked
        """
        waypoints = [np.asarray(wp, dtype=float) for wp in waypoints]
        geometries = []

        # --- Background: dense PLY with original colors, or occupancy grid ---
        occupied = np.argwhere(self.grid)
        world_coords = occupied * self.voxel_size + self.origin
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_coords)
        geometries.append(pcd)

        # --- Waypoints as spheres ---
        for i, wp in enumerate(waypoints):
            voxel = self.world_to_voxel(wp)
            in_obstacle = self.is_occupied(voxel)
            color = [1.0, 0.0, 0.0] if in_obstacle else [0.0, 0.8, 0.0]  # red / green
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(wp)
            sphere.paint_uniform_color(color)
            geometries.append(sphere)

        # --- Paths between waypoints as line sets ---
        for i in range(len(waypoints) - 1):
            p1, p2 = waypoints[i], waypoints[i + 1]
            clear = self.check_segment(p1, p2)
            color = [0.0, 0.8, 0.0] if clear else [1.0, 0.0, 0.0]  # green / red

            # Sample points along segment for visibility
            dist = float(np.linalg.norm(p2 - p1))
            n = max(2, int(dist / 0.05))
            pts = [p1 + t * (p2 - p1) for t in np.linspace(0, 1, n)]
            lines = [[j, j + 1] for j in range(len(pts) - 1)]
            colors = [color] * len(lines)

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(pts)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(line_set)

        print("\nVisualization:")
        for i, wp in enumerate(waypoints):
            voxel = self.world_to_voxel(wp)
            status = "IN OBSTACLE" if self.is_occupied(voxel) else "free"
            print(f"  Waypoint {i}: {wp.tolist()} — {status}")
        for i in range(len(waypoints) - 1):
            status = "BLOCKED" if not self.check_segment(waypoints[i], waypoints[i + 1]) else "clear"
            print(f"  Path {i}→{i+1}: {status}")
        print("\nClose the window to exit.\n")

        o3d.visualization.draw_geometries(
            geometries,
            window_name="Waypoint Mission — Collision Check",
            width=1280,
            height=720,
        )
