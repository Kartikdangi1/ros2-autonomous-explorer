#!/usr/bin/env python3
"""Benchmark script — compare RL agent vs random baseline over N episodes.

Requires the training launch to be running:
    ros2 launch rl_local_planner train.launch.py headless:=true

Then in a separate terminal:
    python3 scripts/benchmark.py [--model rl_best_model/best_model.zip] [--episodes 50]

Output: a formatted table with per-policy metrics.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeMetrics:
    success: bool
    steps: int
    total_reward: float
    collision: bool
    stuck: bool
    timeout: bool
    final_goal_dist: float
    min_scan_seen: float


@dataclass
class PolicyMetrics:
    name: str
    episodes: list[EpisodeMetrics] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def success_rate(self) -> float:
        return sum(e.success for e in self.episodes) / max(self.n, 1)

    @property
    def collision_rate(self) -> float:
        return sum(e.collision for e in self.episodes) / max(self.n, 1)

    @property
    def avg_steps(self) -> float:
        success_episodes = [e for e in self.episodes if e.success]
        if not success_episodes:
            return float('nan')
        return float(np.mean([e.steps for e in success_episodes]))

    @property
    def avg_reward(self) -> float:
        return float(np.mean([e.total_reward for e in self.episodes]))

    @property
    def avg_final_dist(self) -> float:
        return float(np.mean([e.final_goal_dist for e in self.episodes]))


def run_episodes(env, policy, n_episodes: int, policy_name: str) -> PolicyMetrics:
    """Run n_episodes and collect metrics."""
    metrics = PolicyMetrics(name=policy_name)

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        step = 0
        done = False
        min_scan = float('inf')

        while not done:
            if policy is None:
                action = env.action_space.sample()[np.newaxis]
            else:
                action, _ = policy.predict(obs, deterministic=True)

            obs, reward, done, infos = env.step(action)
            info = infos[0]
            total_reward += float(reward[0])
            step += 1
            min_scan = min(min_scan, info.get('min_range', float('inf')))

        metrics.episodes.append(EpisodeMetrics(
            success=bool(info.get('goal_reached', False)),
            steps=step,
            total_reward=total_reward,
            collision=bool(info.get('collision', False)),
            stuck=bool(info.get('stuck', False)),
            timeout=bool(info.get('timeout', False)),
            final_goal_dist=float(info.get('goal_distance', 0)),
            min_scan_seen=min_scan,
        ))

        status = (
            'GOAL' if info.get('goal_reached') else
            'COLL' if info.get('collision') else
            'STUCK' if info.get('stuck') else 'TIME'
        )
        print(f'  [{policy_name}] ep {ep+1:3d}/{n_episodes}: '
              f'{status} | steps={step:3d} | reward={total_reward:+7.2f} | '
              f'dist={info.get("goal_distance", 0):.2f}m')

    return metrics


def print_results_table(results: list[PolicyMetrics]) -> None:
    """Print a formatted comparison table using rich."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print()
    console.rule('[bold]Benchmark Results')

    table = Table(
        'Policy',
        'Episodes',
        'Success rate',
        'Collision rate',
        'Avg steps\n(successful)',
        'Avg reward',
        'Avg final dist (m)',
        box=box.ROUNDED,
        header_style='bold cyan',
        show_lines=True,
    )

    for m in results:
        success_pct = f'{m.success_rate * 100:.1f}%'
        collision_pct = f'{m.collision_rate * 100:.1f}%'
        avg_steps = f'{m.avg_steps:.1f}' if not np.isnan(m.avg_steps) else 'N/A'

        style = 'bold green' if m.success_rate > 0.5 else ''
        table.add_row(
            f'[{style}]{m.name}[/]' if style else m.name,
            str(m.n),
            f'[green]{success_pct}[/]' if m.success_rate > 0.5 else success_pct,
            f'[red]{collision_pct}[/]' if m.collision_rate > 0.3 else collision_pct,
            avg_steps,
            f'{m.avg_reward:+.2f}',
            f'{m.avg_final_dist:.2f}',
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description='Benchmark RL agent vs random baseline')
    parser.add_argument('--model', type=str, default='./rl_best_model/best_model.zip',
                        help='Path to SB3 model checkpoint (.zip)')
    parser.add_argument('--vec-normalize', type=str,
                        default='./rl_checkpoints/vec_normalize.pkl',
                        help='Path to VecNormalize stats (.pkl)')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Episodes per policy')
    parser.add_argument('--stage', type=int, default=1, choices=[0, 1, 2],
                        help='Curriculum stage (0=easy, 1=medium, 2=hard)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip the random-action baseline')
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f'[benchmark] Model not found: {args.model}')
        sys.exit(1)

    import rclpy
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rich.console import Console

    from rl_local_planner.curriculum import CurriculumManager
    from rl_local_planner.gym_env import GazeboExplorerEnv
    from rl_local_planner.reward import RewardWeights

    console = Console()
    console.rule('[bold cyan]RL Local Planner — Benchmark')
    console.print(f'Model:    [green]{args.model}[/]')
    console.print(f'Episodes: [yellow]{args.episodes}[/] per policy')
    console.print(f'Stage:    [yellow]{args.stage}[/]')
    console.print()

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

    all_results = []

    # ── RL policy ─────────────────────────────────────────────────────────
    console.print('[bold]Running RL policy...')
    rl_env = DummyVecEnv([make_env])
    if os.path.isfile(args.vec_normalize):
        rl_env = VecNormalize.load(args.vec_normalize, rl_env)
        rl_env.training = False
        rl_env.norm_reward = False

    rl_model = PPO.load(args.model, env=rl_env, device='cpu')
    rl_results = run_episodes(rl_env, rl_model, args.episodes, 'PPO (RL)')
    all_results.append(rl_results)
    rl_env.close()

    # ── Random baseline ───────────────────────────────────────────────────
    if not args.no_baseline:
        console.print('\n[bold]Running random baseline...')
        rand_env = DummyVecEnv([make_env])
        rand_results = run_episodes(rand_env, None, args.episodes, 'Random')
        all_results.append(rand_results)
        rand_env.close()

    print_results_table(all_results)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
