"""MuJoCo-backed Gymnasium environment for RL local planner training.

Drop-in replacement for GazeboExplorerEnv — identical observation space,
action space, reward function, and curriculum integration.  No ROS2
dependency: the entire physics + sensing pipeline runs inside MuJoCo.

Usage:
    env = MuJoCoExplorerEnv(
        'src/autonomous_explorer/urdf/worlds/mujoco_maze.xml',
        curriculum=CurriculumManager(),
        reward_weights=RewardWeights(),
        seed=42,
    )
    obs, _ = env.reset()
    obs, r, term, trunc, info = env.step(env.action_space.sample())

Expected FPS: 50–200+ (vs ~5 fps in Gazebo).
"""

from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import numpy as np

from rl_local_planner.mujoco_sim import simulate_lidar, lidar_to_costmap
from rl_local_planner.obs_builder import (
    COSTMAP_OBS_SIZE,
    MAX_SCAN_RANGE,
    SCAN_DIM,
    RawSensorData,
    build_observation,
    scale_action,
)
from rl_local_planner.reward import RewardState, RewardWeights, compute_reward
from rl_local_planner.curriculum import CurriculumManager


# ── Constants (match gym_env.py exactly) ─────────────────────────────────────
COLLISION_LIDAR_THRESHOLD = 0.15   # metres — inside chassis = certain collision
STUCK_WINDOW              = 50     # steps
STUCK_THRESHOLD           = 0.05   # metres
COLLISION_GRACE_STEPS     = 5      # steps post-reset where contact check is skipped
FOOTPRINT_HALF_X          = 0.25   # metres
FOOTPRINT_HALF_Y          = 0.15
PHYSICS_TIMESTEP          = 0.05   # 20 Hz — set in mujoco_maze.xml <option>
SPAWN_MIN_CLEARANCE       = 0.30   # metres — LiDAR min after spawn; below = in a wall
MAX_SPAWN_RETRIES         = 10
MAX_GOAL_RETRIES          = 20
GOAL_MIN_CLEARANCE        = 0.20   # metres — ray must reach this close to goal unobstructed


class MuJoCoExplorerEnv(gym.Env):
    """Gymnasium env backed by MuJoCo — same interface as GazeboExplorerEnv."""

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        mjcf_path: str,
        curriculum: CurriculumManager | None = None,
        reward_weights: RewardWeights | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        import mujoco as _mujoco  # imported here so the module loads without mujoco
        self._mujoco = _mujoco
        self._render_mode = render_mode
        self._viewer = None

        self._rng = np.random.default_rng(seed)
        self._curriculum = curriculum or CurriculumManager()
        self._reward_weights = reward_weights or RewardWeights()

        # ── Load model ───────────────────────────────────────────────────
        self._model = _mujoco.MjModel.from_xml_path(mjcf_path)
        self._data  = _mujoco.MjData(self._model)

        # Cache frequently-used IDs
        self._lidar_site_id = _mujoco.mj_name2id(
            self._model, _mujoco.mjtObj.mjOBJ_SITE, 'lidar_site')
        self._robot_body_id = _mujoco.mj_name2id(
            self._model, _mujoco.mjtObj.mjOBJ_BODY, 'base_link')

        # ── Spaces (identical to GazeboExplorerEnv) ──────────────────────
        self.observation_space = gym.spaces.Dict({
            'costmap':     gym.spaces.Box(0, 255, (COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE, 1), np.uint8),
            'scan':        gym.spaces.Box(0.0, 1.0, (SCAN_DIM,), np.float32),
            'goal_vector': gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
            'velocity':    gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

        # ── Episode state ────────────────────────────────────────────────
        self._goal_x: float = 0.0
        self._goal_y: float = 0.0
        self._step_count: int = 0
        self._reward_state  = RewardState()
        self._position_history: deque[tuple[float, float]] = deque(maxlen=STUCK_WINDOW)
        self._dynamics: dict = {}
        self._action_buffer: deque[np.ndarray] = deque()
        self._collision_grace_remaining: int = 0

    # ── Pose helpers ─────────────────────────────────────────────────────

    def _get_yaw(self) -> float:
        """Extract yaw from free-joint quaternion (w, x, y, z at qpos[3:7])."""
        qw = float(self._data.qpos[3])
        qx = float(self._data.qpos[4])
        qy = float(self._data.qpos[5])
        qz = float(self._data.qpos[6])
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny, cosy)

    def _get_robot_state(self) -> tuple[float, float, float, float, float, float]:
        """Return (x, y, yaw, vx_body, vy_body, vyaw) from MuJoCo state.

        MuJoCo free-joint qvel[0:3] is world-frame linear velocity;
        qvel[3:6] is world-frame angular velocity.  We rotate to body frame
        so robot_vx/vy match the body-frame odometry from Gazebo.
        """
        x   = float(self._data.qpos[0])
        y   = float(self._data.qpos[1])
        yaw = self._get_yaw()
        # World-frame velocities
        vx_w = float(self._data.qvel[0])
        vy_w = float(self._data.qvel[1])
        vyaw = float(self._data.qvel[5])
        # Rotate to body frame
        cos_y   = math.cos(yaw)
        sin_y   = math.sin(yaw)
        vx_body =  cos_y * vx_w + sin_y * vy_w
        vy_body = -sin_y * vx_w + cos_y * vy_w
        return x, y, yaw, vx_body, vy_body, vyaw

    # ── Teleport / spawn ─────────────────────────────────────────────────

    def _teleport(self, x: float, y: float, yaw: float = 0.0) -> None:
        """Instantly place the robot at (x, y, yaw) and zero velocity."""
        self._data.qpos[0] = x
        self._data.qpos[1] = y
        self._data.qpos[2] = 0.15           # z — half box height (robot sits on floor)
        self._data.qpos[3] = math.cos(yaw / 2.0)   # qw
        self._data.qpos[4] = 0.0                    # qx
        self._data.qpos[5] = 0.0                    # qy
        self._data.qpos[6] = math.sin(yaw / 2.0)   # qz
        self._data.qvel[:] = 0.0
        self._mujoco.mj_forward(self._model, self._data)

    def _goal_is_reachable(self, spawn_x: float, spawn_y: float,
                           goal_x: float, goal_y: float) -> bool:
        """Return True if a straight-line ray from spawn reaches goal without hitting a wall.

        Uses mj_ray cast from the spawn position toward the goal direction.
        If the ray distance to the nearest geom is >= the spawn-goal distance,
        the path is clear (no wall in the way).
        """
        dx = goal_x - spawn_x
        dy = goal_y - spawn_y
        goal_dist = math.hypot(dx, dy)
        if goal_dist < 1e-6:
            return True
        # Unit direction vector (horizontal ray, z=0)
        direction = np.array([dx / goal_dist, dy / goal_dist, 0.0], dtype=np.float64)
        origin = np.array([spawn_x, spawn_y, 0.15], dtype=np.float64)  # robot centre height
        hit_dist = self._mujoco.mj_ray(
            self._model, self._data,
            origin, direction,
            None,               # geomgroup mask (None = all)
            1,                  # flg_static: include static geoms
            self._robot_body_id,                    # exclude robot body
            np.array([-1], dtype=np.int32),         # geomid output (unused)
        )
        # hit_dist == -1 means no hit (open space); otherwise distance to first geom
        if hit_dist < 0:
            return True
        return hit_dist >= (goal_dist - GOAL_MIN_CLEARANCE)

    def _sample_valid_spawn(self) -> tuple[float, float]:
        """Sample a spawn point not inside a wall (mirrors gym_env.py logic)."""
        candidate = self._curriculum.sample_spawn(self._rng)
        for _ in range(MAX_SPAWN_RETRIES):
            self._teleport(candidate[0], candidate[1])
            scan = simulate_lidar(
                self._model, self._data, self._lidar_site_id,
                bodyexclude=self._robot_body_id,
            )
            if float(np.nanmin(scan)) >= SPAWN_MIN_CLEARANCE:
                return candidate
            candidate = self._curriculum.sample_spawn(self._rng)
        # Exhausted retries — use last candidate anyway so training continues
        self._teleport(candidate[0], candidate[1])
        return candidate

    # ── Gymnasium interface ──────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        spawn = self._sample_valid_spawn()
        # Sample a goal that has clear line-of-sight from the spawn position.
        # Rejects goals inside walls or behind walls (unwinnable episodes).
        goal = self._curriculum.sample_goal(spawn[0], spawn[1], self._rng)
        for _ in range(MAX_GOAL_RETRIES):
            if self._goal_is_reachable(spawn[0], spawn[1], goal[0], goal[1]):
                break
            goal = self._curriculum.sample_goal(spawn[0], spawn[1], self._rng)
        self._goal_x, self._goal_y = goal

        dx = goal[0] - spawn[0]
        dy = goal[1] - spawn[1]
        init_dist = math.hypot(dx, dy)

        self._step_count = 0
        self._reward_state = RewardState(
            initial_goal_distance=max(init_dist, 0.1),
            prev_goal_distance=init_dist,
            prev_action=None,
        )
        self._position_history.clear()
        self._dynamics = self._curriculum.sample_dynamics(self._rng)
        self._action_buffer.clear()
        self._collision_grace_remaining = COLLISION_GRACE_STEPS

        obs = self._get_obs()
        info = {
            'spawn': spawn,
            'goal': goal,
            'initial_distance': init_dist,
            'curriculum_stage': self._curriculum.current_stage,
        }
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        # ── Domain randomization: action delay ───────────────────────────
        delay = self._dynamics.get('action_delay_steps', 0)
        if delay > 0:
            self._action_buffer.append(action.copy())
            if len(self._action_buffer) > delay:
                action = self._action_buffer.popleft()
            else:
                action = np.zeros(3, dtype=np.float32)

        # ── Scale action → physical velocity (body frame) ────────────────
        vx_body, vy_body, vyaw = scale_action(action)
        vel_scale = self._dynamics.get('vel_scale', 1.0)
        vx_body *= vel_scale
        vy_body *= vel_scale
        vyaw    *= vel_scale

        # ── Rotate body-frame → world-frame and write to qvel ────────────
        yaw   = self._get_yaw()
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        self._data.qvel[0] = cos_y * vx_body - sin_y * vy_body   # vx world
        self._data.qvel[1] = sin_y * vx_body + cos_y * vy_body   # vy world
        self._data.qvel[2] = 0.0
        self._data.qvel[3] = 0.0
        self._data.qvel[4] = 0.0
        self._data.qvel[5] = vyaw                                  # yaw rate world

        # ── Physics step ─────────────────────────────────────────────────
        self._mujoco.mj_step(self._model, self._data)

        # ── Read state ───────────────────────────────────────────────────
        x, y, yaw, vx_b, vy_b, vyaw_b = self._get_robot_state()

        # ── Simulate LiDAR ───────────────────────────────────────────────
        scan_metres = simulate_lidar(
            self._model, self._data, self._lidar_site_id,
            bodyexclude=self._robot_body_id,
        )

        # Domain randomization: sensor noise
        scan_noise = self._dynamics.get('scan_noise_sigma', 0.0)
        if scan_noise > 0.0:
            noise = self._rng.normal(0.0, scan_noise, size=scan_metres.shape).astype(np.float32)
            scan_metres = np.clip(scan_metres + noise, 0.0, MAX_SCAN_RANGE)

        # ── Build observation ─────────────────────────────────────────────
        raw = self._build_raw(x, y, yaw, vx_b, vy_b, vyaw_b, scan_metres)
        obs = build_observation(raw)

        # ── Goal distance ─────────────────────────────────────────────────
        goal_dx   = self._goal_x - x
        goal_dy   = self._goal_y - y
        goal_dist = math.hypot(goal_dx, goal_dy)
        min_scan  = float(np.nanmin(scan_metres))

        # ── Termination checks ────────────────────────────────────────────
        goal_reached = goal_dist < self._reward_weights.goal_tolerance
        collision    = self._check_collision(scan_metres)

        self._position_history.append((x, y))
        stuck = False
        if len(self._position_history) >= STUCK_WINDOW:
            oldest = self._position_history[0]
            newest = self._position_history[-1]
            stuck = math.hypot(newest[0] - oldest[0], newest[1] - oldest[1]) < STUCK_THRESHOLD

        timeout    = self._step_count >= self._curriculum.config.max_steps
        terminated = goal_reached or collision
        truncated  = timeout or stuck

        # ── Reward ───────────────────────────────────────────────────────
        reward, reward_info = compute_reward(
            curr_goal_distance=goal_dist,
            min_scan_range=min_scan,
            action=action,
            goal_reached=goal_reached,
            collision=collision,
            state=self._reward_state,
            weights=self._reward_weights,
            robot_yaw=yaw,
            goal_dx=goal_dx,
            goal_dy=goal_dy,
        )

        info = {
            'goal_reached':     goal_reached,
            'collision':        collision,
            'stuck':            stuck,
            'timeout':          timeout,
            'goal_distance':    goal_dist,
            'min_range':        min_scan,
            'step':             self._step_count,
            'curriculum_stage': self._curriculum.current_stage,
            **reward_info,
        }
        return obs, reward, terminated, truncated, info

    # ── Internal helpers ─────────────────────────────────────────────────

    def _check_collision(self, scan_metres: np.ndarray) -> bool:
        """Primary: tight LiDAR threshold.  Secondary: MuJoCo wall contacts.

        Floor contacts (robot body vs worldbody, body id 0) are excluded so
        the robot resting on the floor does not falsely trigger a collision.
        Grace period suppresses the contact check for COLLISION_GRACE_STEPS
        steps after each reset (matches gym_env.py behaviour).
        """
        # 1. LiDAR check — geometry cannot lie
        if float(np.nanmin(scan_metres)) < COLLISION_LIDAR_THRESHOLD:
            return True

        # 2. Contact check — suppressed during grace period
        if self._collision_grace_remaining > 0:
            self._collision_grace_remaining -= 1
            return False

        for i in range(self._data.ncon):
            c    = self._data.contact[i]
            b1   = int(self._model.geom_bodyid[c.geom1])
            b2   = int(self._model.geom_bodyid[c.geom2])
            if b1 == self._robot_body_id or b2 == self._robot_body_id:
                other = b2 if b1 == self._robot_body_id else b1
                if other != 0:   # 0 = worldbody (floor); non-zero = wall/pillar
                    return True
        return False

    def _build_raw(
        self,
        x: float, y: float, yaw: float,
        vx: float, vy: float, vyaw: float,
        scan_metres: np.ndarray,
    ) -> RawSensorData:
        """Populate RawSensorData from MuJoCo state for obs_builder."""
        raw = RawSensorData()
        raw.costmap        = lidar_to_costmap(scan_metres)
        raw.costmap_width  = COSTMAP_OBS_SIZE   # already 84×84
        raw.costmap_height = COSTMAP_OBS_SIZE
        raw.costmap_resolution = 0.12
        raw.scan_ranges    = scan_metres
        raw.robot_x        = x
        raw.robot_y        = y
        raw.robot_yaw      = yaw
        raw.robot_vx       = vx
        raw.robot_vy       = vy
        raw.robot_vyaw     = vyaw
        raw.goal_x         = self._goal_x
        raw.goal_y         = self._goal_y
        return raw

    def _get_obs(self) -> dict[str, np.ndarray]:
        x, y, yaw, vx, vy, vyaw = self._get_robot_state()
        scan = simulate_lidar(
            self._model, self._data, self._lidar_site_id,
            bodyexclude=self._robot_body_id,
        )
        return build_observation(self._build_raw(x, y, yaw, vx, vy, vyaw, scan))

    def render(self) -> None:
        """Open / update the MuJoCo passive viewer window.

        Call this after every env.step() to show the simulation in real time.
        A green sphere is drawn at the current goal position so you can see
        where the robot is headed.
        """
        if self._render_mode != 'human':
            return

        import mujoco.viewer as _mj_viewer

        # Lazily open the viewer on first call
        if self._viewer is None:
            self._viewer = _mj_viewer.launch_passive(self._model, self._data)

        # If the user closed the window, stop rendering quietly
        if not self._viewer.is_running():
            self._viewer = None
            self._render_mode = None
            return

        # Draw goal marker in the user scene
        with self._viewer.lock():
            scn = self._viewer.user_scn
            scn.ngeom = 0
            if scn.maxgeom > 0:
                g = scn.geoms[0]
                self._mujoco.mjv_initGeom(
                    g,
                    self._mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.zeros(3),
                    np.array([self._goal_x, self._goal_y, 0.2], dtype=np.float64),
                    np.eye(3).flatten(),
                    np.array([0.0, 1.0, 0.0, 0.8], dtype=np.float32),
                )
                g.size[:] = [0.2, 0.2, 0.2]
                scn.ngeom = 1

        self._viewer.sync()

    def set_curriculum_stage(self, stage: int) -> None:
        """Propagate curriculum stage from the main process to this worker.

        Called by CurriculumCallback via SubprocVecEnv.env_method() so that
        all worker processes advance together when the main-process curriculum
        decides to move to the next stage.
        """
        self._curriculum.current_stage = stage
        self._curriculum._successes.clear()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
