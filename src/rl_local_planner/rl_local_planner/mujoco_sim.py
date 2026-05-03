"""MuJoCo physics helpers: software LiDAR and costmap generation.

Used by mujoco_env.py — no ROS2 dependencies.
"""

from __future__ import annotations

import cv2
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

    # Pre-compute all ray directions in world frame — eliminates 360 per-ray
    # np.array() allocations and trig calls from the original loop.
    angles = 2.0 * np.pi * np.arange(n_rays) / n_rays
    dirs_body = np.column_stack(
        [np.cos(angles), np.sin(angles), np.zeros(n_rays)])  # (360, 3)
    dirs_world = (mat @ dirs_body.T).T                        # (360, 3) batch rotation

    scan = np.full(n_rays, max_range, dtype=np.float32)
    geomid = np.array([-1], dtype=np.int32)

    for i in range(n_rays):
        dist = mujoco.mj_ray(
            model, data, pos, dirs_world[i],
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
    resolution: float = 0.05,
    size: int = 84,
    inflation_cells: int = 4,
) -> np.ndarray:
    """Convert a 360-ray lidar scan to a robot-centred occupancy grid.

    Produces the same value conventions as the Nav2 local costmap so that
    obs_builder.build_costmap_obs() works without modification:
      0   = free
      253 = inscribed (inflated obstacle — robot centre would be in collision)
      254 = lethal (obstacle cell)

    The grid is 84×84 cells at 0.05 m/cell → ±2.1 m view radius around robot.
    inflation_cells=4 → 0.20 m inflation radius, matching nav2 inflation_layer.

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

    # --- mark lethal cells (vectorised) ---------------------------------------
    angles = 2.0 * np.pi * np.arange(n_rays) / n_rays
    x_m = scan_metres * np.cos(angles)
    y_m = scan_metres * np.sin(angles)
    cols = (center + x_m / resolution).astype(np.int32)
    rows = (center - y_m / resolution).astype(np.int32)
    valid = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    lethal_rows = rows[valid]
    lethal_cols = cols[valid]
    grid[lethal_rows, lethal_cols] = 254

    # --- inflate lethal cells → inscribed (253) via cv2.dilate ---------------
    # cv2.dilate with an elliptical kernel is ~40× faster than scipy.binary_dilation
    # and ~1500× faster than the original triple Python for-loop.
    if lethal_rows.size > 0:
        kernel_size = 2 * inflation_cells + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        lethal_mask = (grid == 254).astype(np.uint8)
        inscribed_mask = cv2.dilate(lethal_mask, kernel)
        grid[inscribed_mask.astype(bool)] = 253
        # Re-stamp lethal cells — dilation may have overwritten them with 253
        grid[lethal_rows, lethal_cols] = 254

    return grid.flatten()
