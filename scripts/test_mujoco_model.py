#!/usr/bin/env python3
"""
Test a trained PPO model on the MuJoCo environment.

Usage:
    python3 scripts/test_mujoco_model.py
    python3 scripts/test_mujoco_model.py --model ./rl_checkpoints/ppo_final \
        --vec-normalize ./rl_checkpoints/vec_normalize.pkl --stage 3 --episodes 20
    python3 scripts/test_mujoco_model.py --render --delay 0.05
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rl_local_planner.mujoco_env import MuJoCoExplorerEnv
from rl_local_planner.curriculum import CurriculumManager
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage


def main():
    parser = argparse.ArgumentParser(description='Test a trained PPO model on MuJoCo environment')
    parser.add_argument(
        '--model',
        type=str,
        default='./rl_checkpoints/ppo_final',
        help='Path to trained model (without .zip)',
    )
    parser.add_argument(
        '--vec-normalize',
        type=str,
        default='./rl_checkpoints/vec_normalize.pkl',
        help='Path to VecNormalize stats pickle (set to "" to skip)',
    )
    parser.add_argument(
        '--stage',
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help='Curriculum stage (0=bootstrap, 1=easy, 2=medium, 3=hard)',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Number of test episodes',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Override episode step limit (default: curriculum stage default)',
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Open MuJoCo viewer window to watch the simulation',
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.05,
        help='Seconds to sleep between steps when rendering (default: 0.05 = real-time)',
    )
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    stage_names = {0: 'bootstrap (0.3-0.8m)', 1: 'easy (0.8-2m)', 2: 'medium (2-4m)', 3: 'hard (3-6m)'}
    stage_name = stage_names[args.stage]

    # ── Build curriculum ──────────────────────────────────────────────────────
    curriculum = CurriculumManager()
    curriculum.current_stage = args.stage
    if args.max_steps is not None:
        curriculum.config.max_steps = args.max_steps

    render_mode = 'human' if args.render else None

    # ── Build env — must mirror training wrapper stack exactly ────────────────
    # Training used: DummyVecEnv → VecTransposeImage → VecNormalize(norm_obs=False)
    # VecTransposeImage converts costmap (H,W,C)→(C,H,W) so the CNN sees channels-first.
    # Skipping it causes a shape mismatch in RobotFeatureExtractor.
    print(f"Creating MuJoCo environment (Stage {args.stage}: {stage_name}, "
          f"max_steps={curriculum.config.max_steps})...")

    def _make():
        return MuJoCoExplorerEnv(
            'src/autonomous_explorer/urdf/worlds/mujoco_maze.xml',
            curriculum=curriculum,
            render_mode=render_mode,
            seed=args.seed,
        )

    vec_env = DummyVecEnv([_make])
    vec_env = VecTransposeImage(vec_env)

    if args.vec_normalize and Path(args.vec_normalize).is_file():
        print(f"Loading VecNormalize stats from {args.vec_normalize}")
        vec_env = VecNormalize.load(args.vec_normalize, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("No VecNormalize stats — evaluating without reward normalisation")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = args.model.rstrip('.zip')
    print(f"\nLoading model from {model_path}.zip...")
    try:
        model = PPO.load(model_path, env=vec_env)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    # ── Run episodes ──────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"Testing PPO Model — Stage {args.stage} ({stage_name})")
    print("=" * 75 + "\n")

    n_goal = n_collision = n_stuck = n_timeout = 0
    ep_rewards: list[float] = []
    ep_steps:   list[int]   = []
    ep_dists:   list[float] = []

    obs = vec_env.reset()
    ep_reward = 0.0
    ep_step   = 0
    ep_done   = 0

    while ep_done < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        if args.render:
            vec_env.env_method('render')
            time.sleep(args.delay)

        ep_reward += float(reward[0])
        ep_step   += 1

        if done[0]:
            ep_done += 1
            i = info[0]
            goal_reached   = i.get('goal_reached', False)
            collision      = i.get('collision', False)
            stuck          = i.get('stuck', False)
            timeout        = i.get('timeout', False)
            final_distance = i.get('goal_distance', float('nan'))
            min_lidar      = i.get('min_range', float('nan'))

            n_goal      += int(goal_reached)
            n_collision += int(collision)
            n_stuck     += int(stuck)
            n_timeout   += int(timeout)

            ep_rewards.append(ep_reward)
            ep_steps.append(ep_step)
            ep_dists.append(final_distance)

            outcome = (
                'SUCCESS  ' if goal_reached else
                'COLLISION' if collision     else
                'STUCK    ' if stuck         else
                'TIMEOUT  '
            )
            print(f"Episode {ep_done:3d}/{args.episodes}: {outcome} | "
                  f"Steps: {ep_step:4d} | Reward: {ep_reward:8.2f} | "
                  f"Dist: {final_distance:5.2f}m | MinLiDAR: {min_lidar:5.2f}m")

            ep_reward = 0.0
            ep_step   = 0
            obs = vec_env.reset()

    # ── Summary ───────────────────────────────────────────────────────────────
    n = args.episodes
    print("\n" + "=" * 75)
    print(f"Summary  ({n} episodes, stage {args.stage}: {stage_name})")
    print("=" * 75)
    print(f"  Goal reached : {n_goal:3d}/{n}  ({100*n_goal/n:.1f}%)")
    print(f"  Collision    : {n_collision:3d}/{n}  ({100*n_collision/n:.1f}%)")
    print(f"  Stuck        : {n_stuck:3d}/{n}  ({100*n_stuck/n:.1f}%)")
    print(f"  Timeout      : {n_timeout:3d}/{n}  ({100*n_timeout/n:.1f}%)")
    print(f"  Mean reward  : {np.mean(ep_rewards):.2f} ± {np.std(ep_rewards):.2f}")
    print(f"  Mean steps   : {np.mean(ep_steps):.1f}")
    print(f"  Mean final dist : {np.nanmean(ep_dists):.2f}m")
    print("=" * 75 + "\n")

    # Close the viewer before tearing down the MuJoCo model to avoid
    # a GLFW/MuJoCo double-free crash on exit when --render is used.
    if args.render:
        vec_env.env_method('close')
    vec_env.close()


if __name__ == '__main__':
    main()
