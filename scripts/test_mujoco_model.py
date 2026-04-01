#!/usr/bin/env python3
"""
Test a trained PPO model on the MuJoCo environment.

Usage:
    python3 scripts/test_mujoco_model.py
    python3 scripts/test_mujoco_model.py --model ./rl_checkpoints/ppo_final.zip --stage 2 --episodes 5
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rl_local_planner.mujoco_env import MuJoCoExplorerEnv
from rl_local_planner.curriculum import CurriculumManager
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser(description='Test a trained PPO model on MuJoCo environment')
    parser.add_argument(
        '--model',
        type=str,
        default='./rl_checkpoints/ppo_final.zip',
        help='Path to trained model'
    )
    parser.add_argument(
        '--stage',
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help='Curriculum stage (0=bootstrap, 1=easy, 2=medium, 3=hard)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of test episodes'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=True,
        help='Use deterministic policy (no randomness)'
    )
    args = parser.parse_args()

    # Load model
    print(f"\nLoading model from {args.model}...")
    try:
        model = PPO.load(args.model)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    # Create curriculum manager
    curriculum = CurriculumManager()
    curriculum.current_stage = args.stage

    stage_names = {0: 'bootstrap (0.3-0.8m)', 1: 'easy (0.8-2m)', 2: 'medium (2-4m)', 3: 'hard (3-6m)'}
    stage_name = stage_names.get(args.stage, 'unknown')

    # Create environment
    print(f"Creating MuJoCo environment (Stage {args.stage}: {stage_name})...")
    env = MuJoCoExplorerEnv(
        'src/autonomous_explorer/urdf/worlds/mujoco_maze.xml',
        curriculum=curriculum
    )

    # Run episodes
    print("\n" + "="*75)
    print(f"Testing PPO Model — Stage {args.stage} ({stage_name})")
    print("="*75 + "\n")

    successes = 0
    total_rewards = []
    final_distances = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
            total_reward += r

        goal_reached = info.get('goal_reached', False)
        final_distance = info.get('goal_distance', 0.0)
        min_lidar = info.get('min_range', 0.0)

        if goal_reached:
            successes += 1
            result = "✓ SUCCESS"
        else:
            result = "✗ FAIL  "

        total_rewards.append(total_reward)
        final_distances.append(final_distance)

        print(f"Episode {ep+1:2d}: {result} | Steps: {steps:3d} | Reward: {total_reward:7.2f} | "
              f"Distance: {final_distance:5.2f}m | Min LiDAR: {min_lidar:5.2f}m")

    # Summary
    print("\n" + "="*75)
    print(f"Summary: {successes}/{args.episodes} successes ({100*successes/args.episodes:.0f}%)")
    print(f"  Avg reward:       {sum(total_rewards)/len(total_rewards):7.2f}")
    print(f"  Avg final dist:   {sum(final_distances)/len(final_distances):7.2f}m")
    print(f"  Min final dist:   {min(final_distances):7.2f}m")
    print("="*75 + "\n")

    env.close()


if __name__ == '__main__':
    main()
