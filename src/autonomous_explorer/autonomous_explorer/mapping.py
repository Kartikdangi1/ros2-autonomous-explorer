"""
mapping.py
==========
OccupancyMapManager — maintains the OccupancyMapper that mirrors
the SLAM Toolbox /map topic.

Separating this from the planner means you can tune or swap the map
synchronisation logic without touching NBV planning code.

Usage
-----
from autonomous_explorer.mapping import OccupancyMapManager

mgr = OccupancyMapManager(node.get_logger())
# In /map callback:
mgr.update_from_slam(msg)
# In planning code:
mapper = mgr.mapper   # may be None before first /map arrives
"""

import threading

import numpy as np
from nav_msgs.msg import OccupancyGrid

from autonomous_explorer.nbv_utils import OccupancyMapper


class OccupancyMapManager:
    """
    Thread-safe wrapper around OccupancyMapper that stays in sync with
    SLAM Toolbox /map messages.

    Key behaviours:
    - Recreates (and resizes) the mapper when the SLAM map dimensions or
      origin change, preserving any existing log-odds data in the overlap.
    - Applies SLAM's occupied/free labels while leaving SLAM-unknown cells
      at their current log-odds value so scan-based obstacle data is not erased.
    """

    def __init__(self, logger):
        self._mapper: OccupancyMapper | None = None
        self._lock = threading.Lock()
        self._log  = logger

    # ── Public interface ───────────────────────────────────────────────────────

    @property
    def mapper(self) -> OccupancyMapper | None:
        """Current mapper, or None before the first /map message arrives."""
        with self._lock:
            return self._mapper

    def update_from_slam(self, msg: OccupancyGrid) -> None:
        """
        Sync the internal OccupancyMapper from a SLAM Toolbox /map message.

        Call this directly from the node's /map subscription callback.
        """
        w   = msg.info.width
        h   = msg.info.height
        res = msg.info.resolution
        ox  = msg.info.origin.position.x
        oy  = msg.info.origin.position.y

        with self._lock:
            if (self._mapper is None
                    or self._mapper.width    != w
                    or self._mapper.height   != h
                    or self._mapper.origin_x != ox
                    or self._mapper.origin_y != oy):
                self._mapper = self._resize(w, h, res, ox, oy)
                self._log.info(
                    f'OccupancyMapper resized: {w}×{h} @ {res:.3f} m/cell '
                    f'origin=({ox:.1f}, {oy:.1f})')

            data = np.array(msg.data, dtype=np.int8).reshape(h, w)
            with self._mapper._lock:
                lo = self._mapper.log_odds
                lo[data == 100] =  5.0   # occupied
                lo[data ==   0] = -2.0   # free
                # SLAM-unknown (-1): leave existing log-odds intact so that
                # scan-based obstacle data in unexplored cells is preserved
                # across map updates.
                self._mapper._map_version += 1

    # ── Private ────────────────────────────────────────────────────────────────

    def _resize(self, w: int, h: int, res: float,
                ox: float, oy: float) -> OccupancyMapper:
        """
        Create a new OccupancyMapper and copy overlapping log-odds data from
        the old mapper so accumulated obstacle history is not lost when SLAM
        expands the map.
        """
        new_mapper = OccupancyMapper(width=w, height=h, resolution=res,
                                     origin_x=ox, origin_y=oy)
        if self._mapper is None:
            return new_mapper

        # Cell offset of old mapper's origin in the new grid
        dx = int(round((self._mapper.origin_x - ox) / res))
        dy = int(round((self._mapper.origin_y - oy) / res))

        src_x0 = max(0, -dx);  src_y0 = max(0, -dy)
        dst_x0 = max(0,  dx);  dst_y0 = max(0,  dy)
        copy_w  = min(self._mapper.width  - src_x0, w - dst_x0)
        copy_h  = min(self._mapper.height - src_y0, h - dst_y0)

        if copy_w > 0 and copy_h > 0:
            with self._mapper._lock:
                src = self._mapper.log_odds[
                    src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
                new_mapper.log_odds[
                    dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = src

        return new_mapper
