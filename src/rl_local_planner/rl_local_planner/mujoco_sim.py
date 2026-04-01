"""MuJoCo physics helpers: software LiDAR and costmap generation.

Used by mujoco_env.py — no ROS2 dependencies.
"""

from __future__ import annotations

import numpy as np


def simulate_lidar(
    model,
    data,
    site_id: int,
    n_rays: int = 360,
    max_range: float = 18.0,
    bodyexclude: int = -1,
) -> np.ndarray:
    """Cast n_rays horizontal rays from the lidar site, return distances (metres).

    Uses mujoco.mj_ray() for each angle.  Rays that miss all geometry return
    max_range (matching NaN → max_range semantics in the Gazebo env).

    Args:
        model: MjModel
        data:  MjData (must be after mj_forward / mj_step)
        site_id: index of the lidar site (from mj_name2id)
        n_rays: number of rays (default 360 → one per degree)
        max_range: maximum sensor range in metres
        bodyexclude: body id to exclude from ray intersections (set to robot body
                     id so the robot doesn't hit itself)

    Returns:
        ndarray of shape (n_rays,) float32, values in [0, max_range]
    """
    import mujoco  # local import so module loads without mujoco at import time

    # Lidar site world position and rotation matrix
    pos = data.site_xpos[site_id].copy()          # (3,) world position
    mat = data.site_xmat[site_id].reshape(3, 3)   # world rotation matrix

    scan = np.full(n_rays, max_range, dtype=np.float32)
    geomid = np.array([-1], dtype=np.int32)

    for i in range(n_rays):
        angle = 2.0 * np.pi * i / n_rays
        # Horizontal ray in body frame (z=0 plane)
        dir_body = np.array([np.cos(angle), np.sin(angle), 0.0])
        # Rotate to world frame
        dir_world = mat @ dir_body
        dist = mujoco.mj_ray(
            model, data, pos, dir_world,
            None,           # geomgroup bitmask — include all
            1,              # flg_static — include static geoms
            bodyexclude,    # exclude robot body so ray doesn't self-intersect
            geomid,
        )
        if 0.0 <= dist < max_range:
            scan[i] = float(dist)
        # else: leave at max_range (no hit or hit beyond range)

    return scan


def lidar_to_costmap(
    scan_metres: np.ndarray,
    resolution: float = 0.12,
    size: int = 84,
    inflation_cells: int = 2,
) -> np.ndarray:
    """Convert a 360-ray lidar scan to a robot-centred occupancy grid.

    Produces the same value conventions as the Nav2 local costmap so that
    obs_builder.build_costmap_obs() works without modification:
      0   = free
      253 = inscribed (inflated obstacle — robot centre would be in collision)
      254 = lethal (obstacle cell)

    The grid is 84×84 cells at 0.12 m/cell → ±5.04 m view radius around robot.

    Args:
        scan_metres: (N,) array of lidar ranges in metres (world scale)
        resolution:  metres per cell
        size:        grid side length in cells (84)
        inflation_cells: how many cells to inflate around each lethal cell

    Returns:
        Flat uint8 array of length size*size, row-major (compatible with
        RawSensorData.costmap which is reshaped via costmap_height × costmap_width).
    """
    grid = np.zeros((size, size), dtype=np.uint8)
    center = size // 2          # robot sits at this cell
    n_rays = len(scan_metres)

    # --- mark lethal cells ---------------------------------------------------
    lethal_cells: list[tuple[int, int]] = []
    for i, dist in enumerate(scan_metres):
        angle = 2.0 * np.pi * i / n_rays
        x_m = dist * np.cos(angle)
        y_m = dist * np.sin(angle)
        # image convention: col increases right (+x), row increases down (-y)
        col = int(center + x_m / resolution)
        row = int(center - y_m / resolution)
        if 0 <= row < size and 0 <= col < size:
            grid[row, col] = 254
            lethal_cells.append((row, col))

    # --- inflate lethal cells → inscribed (253) ------------------------------
    for r, c in lethal_cells:
        for dr in range(-inflation_cells, inflation_cells + 1):
            for dc in range(-inflation_cells, inflation_cells + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] < 253:
                    grid[nr, nc] = 253

    # Re-stamp lethal cells in case inflation overwrote them with 253
    for r, c in lethal_cells:
        grid[r, c] = 254

    return grid.flatten()
