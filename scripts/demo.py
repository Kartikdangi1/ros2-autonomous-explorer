#!/usr/bin/env python3
"""Demo script — run a single episode with the trained RL agent.

Requires the training launch to be running:
    ros2 launch rl_local_planner train.launch.py headless:=false use_rviz:=true

Then in a separate terminal:
    python3 scripts/demo.py [--model rl_best_model/best_model.zip]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import rclpy


def main():
    parser = argparse.ArgumentParser(description='Run a demo episode with the trained RL agent')
    parser.add_argument(
        '--model', type=str, default='./rl_best_model/best_model.zip',
        help='Path to SB3 model checkpoint (.zip)')
    parser.add_argument(
        '--vec-normalize', type=str, default='./rl_checkpoints/vec_normalize.pkl',
        help='Path to VecNormalize stats (.pkl) — optional')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Episode seed')
    parser.add_argument(
        '--stage', type=int, default=2, choices=[0, 1, 2],
        help='Curriculum stage for the demo episode (0=easy, 1=medium, 2=hard)')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f'[demo] Model not found: {args.model}')
        print('       Train a model first with train_ppo.py, or pass --model <path>')
        sys.exit(1)

    # Imports after arg parsing so --help is fast
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rich.console import Console
    from rich.table import Table
    from rich import box

    from rl_local_planner.curriculum import CurriculumManager
    from rl_local_planner.gym_env import GazeboExplorerEnv
    from rl_local_planner.reward import RewardWeights

    console = Console()
    console.rule('[bold cyan]RL Local Planner — Demo Episode')
    console.print(f'Model:  [green]{args.model}[/]')
    console.print(f'Stage:  [yellow]{args.stage}[/] (0=easy 1=medium 2=hard)')
    console.print('')

    if not rclpy.ok():
        rclpy.init()

    curriculum = CurriculumManager()
    curriculum.current_stage = args.stage

    def make_env():
        return GazeboExplorerEnv(
            curriculum=curriculum,
            reward_weights=RewardWeights(),
            seed=args.seed,
        )

    env = DummyVecEnv([make_env])

    # Load VecNormalize stats if available
    if os.path.isfile(args.vec_normalize):
        env = VecNormalize.load(args.vec_normalize, env)
        env.training = False
        env.norm_reward = False
        console.print(f'VecNormalize stats loaded from [dim]{args.vec_normalize}[/]')

    model = PPO.load(args.model, env=env, device='cpu')
    console.print(f'Policy loaded: [bold]{type(model.policy).__name__}[/]\n')

    obs = env.reset()
    total_reward = 0.0
    step = 0

    table = Table(
        'Step', 'vx', 'vy', 'vyaw',
        'Goal dist (m)', 'Min scan (m)', 'Reward', 'Cumulative',
        box=box.SIMPLE_HEAVY, show_header=True, header_style='bold magenta',
    )

    done = False
    info = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        info = infos[0]
        step += 1
        total_reward += float(reward[0])

        a = action[0]
        table.add_row(
            str(step),
            f'{a[0]:.3f}',
            f'{a[1]:.3f}',
            f'{a[2]:.3f}',
            f'{info.get("goal_distance", 0):.2f}',
            f'{info.get("min_range", 0):.2f}',
            f'{float(reward[0]):.3f}',
            f'{total_reward:.3f}',
        )

    console.print(table)
    console.rule('[bold]Episode Complete')

    outcome = (
        '[bold green]Goal reached![/]' if info.get('goal_reached') else
        '[bold red]Collision![/]' if info.get('collision') else
        '[bold yellow]Stuck[/]' if info.get('stuck') else
        '[bold yellow]Timeout[/]'
    )
    console.print(f'Outcome:         {outcome}')
    console.print(f'Steps taken:     [cyan]{step}[/]')
    console.print(f'Total reward:    [cyan]{total_reward:.3f}[/]')
    console.print(f'Final goal dist: [cyan]{info.get("goal_distance", 0):.2f} m[/]')

    env.close()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
