#!/usr/bin/env python3
"""Optuna hyperparameter sweep for the RL local planner.

Wraps the PPO training logic in an Optuna objective function and optimises
over learning rate, rollout length, batch size, entropy coefficient, and
reward weights.

Usage:
    # Start the training launch first, then:
    python3 sweep_optuna.py [--trials 20] [--study-name my_sweep]

Results are stored in a local SQLite DB (sweep_results.db) so crashed
trials can be resumed automatically.
"""

from __future__ import annotations

import argparse
import os

import optuna
import rclpy
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_local_planner.config_schema import TrainingConfig
from rl_local_planner.curriculum import CurriculumManager
from rl_local_planner.feature_extractor import RobotFeatureExtractor
from rl_local_planner.gym_env import GazeboExplorerEnv
from rl_local_planner.reward import RewardWeights


def make_env(curriculum: CurriculumManager, reward_weights: RewardWeights, seed: int):
    def _init():
        return GazeboExplorerEnv(curriculum=curriculum, reward_weights=reward_weights, seed=seed)
    return _init


def objective(trial: optuna.Trial) -> float:
    """Train PPO with sampled hyperparameters; return best eval mean reward."""

    # ── Sample hyperparameters ────────────────────────────────────────────
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_epochs = trial.suggest_int('n_epochs', 3, 20)
    ent_coef = trial.suggest_float('ent_coef', 1e-4, 0.1, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)

    # Reward weight sweeps
    reward_progress = trial.suggest_float('reward_progress', 1.0, 10.0)
    reward_goal_reached = trial.suggest_float('reward_goal_reached', 5.0, 20.0)
    reward_collision = trial.suggest_float('reward_collision', -10.0, -1.0)
    reward_proximity = trial.suggest_float('reward_proximity', -2.0, 0.0)

    seed = trial.number * 1000 + 42

    cfg = TrainingConfig(
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        total_timesteps=100_000,  # shorter for sweep trials
        eval_freq=10_000,
        n_eval_episodes=5,
        reward_progress=reward_progress,
        reward_goal_reached=reward_goal_reached,
        reward_collision=reward_collision,
        reward_proximity=reward_proximity,
        tb_log_dir=f'./tb_logs/sweep/trial_{trial.number}',
        save_dir=f'./rl_checkpoints/sweep/trial_{trial.number}',
        best_model_dir=f'./rl_best_model/sweep/trial_{trial.number}',
    )

    reward_weights = RewardWeights(
        progress=cfg.reward_progress,
        goal_reached=cfg.reward_goal_reached,
        collision=cfg.reward_collision,
        proximity=cfg.reward_proximity,
    )
    curriculum = CurriculumManager()

    env = DummyVecEnv([make_env(curriculum, reward_weights, seed)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=cfg.gamma)
    eval_env = DummyVecEnv([make_env(curriculum, reward_weights, seed + 500)])

    policy_kwargs = {
        'features_extractor_class': RobotFeatureExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.best_model_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.best_model_dir,
        log_path=cfg.tb_log_dir,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        verbose=0,
    )

    model = PPO(
        'MultiInputPolicy',
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=cfg.tb_log_dir,
        policy_kwargs=policy_kwargs,
        seed=seed,
    )

    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=eval_callback)
    except Exception as e:
        print(f'Trial {trial.number} failed: {e}')
        return float('-inf')
    finally:
        env.close()
        eval_env.close()

    # Return best eval reward (stored by EvalCallback in evaluations.npz)
    import numpy as np
    eval_file = os.path.join(cfg.tb_log_dir, 'evaluations.npz')
    if os.path.isfile(eval_file):
        data = np.load(eval_file)
        return float(np.max(data['results'].mean(axis=1)))
    return float('-inf')


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter sweep for PPO')
    parser.add_argument('--trials', type=int, default=20,
                        help='Number of Optuna trials')
    parser.add_argument('--study-name', type=str, default='ppo_sweep',
                        help='Optuna study name')
    parser.add_argument('--storage', type=str, default='sqlite:///sweep_results.db',
                        help='Optuna storage URL (SQLite by default for resumability)')
    parser.add_argument('--direction', type=str, default='maximize',
                        choices=['maximize', 'minimize'])
    args = parser.parse_args()

    if not rclpy.ok():
        rclpy.init()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=args.direction,
        load_if_exists=True,  # resume crashed sweeps
    )

    print(f'Starting Optuna sweep: {args.trials} trials')
    print(f'Storage: {args.storage}')
    print(f'Visualise: optuna-dashboard {args.storage}')

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print('\n=== Best trial ===')
    best = study.best_trial
    print(f'  Value: {best.value:.4f}')
    print('  Params:')
    for k, v in best.params.items():
        print(f'    {k}: {v}')

    # Save best params as YAML
    best_cfg_path = 'best_sweep_config.yaml'
    with open(best_cfg_path, 'w') as f:
        yaml.dump(best.params, f, default_flow_style=False)
    print(f'\nBest config saved to: {best_cfg_path}')

    rclpy.shutdown()


if __name__ == '__main__':
    main()
