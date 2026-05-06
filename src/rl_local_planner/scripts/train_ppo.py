#!/usr/bin/env python3
"""PPO training script for the RL local planner.

Usage:
    python3 train_ppo.py                        # defaults (mujoco, seed=42)
    python3 train_ppo.py sim=mujoco num_envs=8  # Hydra overrides
    python3 train_ppo.py reward_proximity=-1.0  # override any param
    python3 train_ppo.py --cfg job              # print resolved config, no training

Hydra saves the full resolved config to outputs/<date>/<time>/.hydra/config.yaml.
A training_config_used.yaml is also saved alongside the final model checkpoint.

TensorBoard:
    tensorboard --logdir ./tb_logs/
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import uuid

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecTransposeImage

from rl_local_planner.config_schema import TrainingConfig
from rl_local_planner.curriculum import CurriculumManager
from rl_local_planner.feature_extractor import RobotFeatureExtractor
from rl_local_planner.reward import RewardWeights


# ── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(log_dir: str, run_id: str) -> logging.Logger:
    """Configure structured logging with file + console handlers."""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('rl_training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = f'[%(asctime)s] [%(name)s] [run:{run_id}] %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    log_file = os.path.join(log_dir, f'training_{run_id}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ── Seeding ──────────────────────────────────────────────────────────────────

def seed_everything(seed: int, logger: logging.Logger) -> None:
    """Set seeds for NumPy, PyTorch, and CUDA for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True)
        logger.info('Deterministic algorithms enabled')
    except RuntimeError as e:
        logger.warning('Could not enable deterministic algorithms: %s', e)

    logger.info('Seeds set: numpy=%d, torch=%d', seed, seed)


# ── Model card ───────────────────────────────────────────────────────────────

def save_model_card(
    save_path: str,
    cfg: TrainingConfig,
    seed: int,
    curriculum_stage: int,
    total_timesteps_done: int = 0,
) -> None:
    """Write a model_card.yaml alongside a saved checkpoint."""
    # Git commit
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = 'unknown'

    # Config hash
    config_dict = cfg.model_dump()
    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    card = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_commit': git_commit,
        'training_config_hash': config_hash,
        'seed': seed,
        'total_timesteps': total_timesteps_done,
        'curriculum_stage_reached': curriculum_stage,
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'sb3_version': '2.7.1',
        'training_config': config_dict,
    }

    card_path = save_path.rstrip('.zip') + '_model_card.yaml'
    with open(card_path, 'w') as f:
        yaml.dump(card, f, default_flow_style=False, sort_keys=False)


# ── Callbacks ────────────────────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """Logs per-step training metrics to TensorBoard."""

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        for info in infos:
            # Per-step: valid at every step
            if 'min_range' in info:
                self.logger.record('safety/min_lidar_range', info['min_range'])
            if 'curriculum_stage' in info:
                self.logger.record('curriculum/stage', info['curriculum_stage'])
            # Terminal-only: only meaningful at episode end
            is_terminal = (
                info.get('goal_reached', False) or info.get('collision', False)
                or info.get('timeout', False) or info.get('stuck', False)
            )
            if is_terminal:
                if 'collision' in info:
                    self.logger.record('episode/collision', float(info['collision']))
                if 'goal_reached' in info:
                    self.logger.record('episode/goal_reached', float(info['goal_reached']))
                if 'goal_distance' in info:
                    self.logger.record('episode/goal_distance', info['goal_distance'])
        return True


class CurriculumCallback(BaseCallback):
    """Advances curriculum stage based on evaluation success rate."""

    def __init__(self, curriculum: CurriculumManager, verbose: int = 0):
        super().__init__(verbose)
        self._curriculum = curriculum

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        for info in infos:
            # Record terminal episodes only
            if info.get('goal_reached', False) or info.get('collision', False) \
                    or info.get('timeout', False) or info.get('stuck', False):
                self._curriculum.record_episode(info.get('goal_reached', False))

        if self._curriculum.maybe_advance():
            new_stage = self._curriculum.current_stage
            self.logger.record('curriculum/stage', new_stage)
            # Propagate new stage to all worker environments (needed for
            # SubprocVecEnv where each worker has a pickled copy of curriculum)
            self.training_env.env_method('set_curriculum_stage', new_stage)
            # Reset VecNormalize reward running stats so Stage N's short-episode
            # distribution does not corrupt Stage N+1 normalization.
            if isinstance(self.training_env, VecNormalize):
                self.training_env.ret_rms.mean = 0.0
                self.training_env.ret_rms.var = 1.0
                self.training_env.ret_rms.count = 1e-4
                self.training_env.returns = np.zeros(self.training_env.num_envs)
        return True


class ModelCardCallback(BaseCallback):
    """Saves a model_card.yaml alongside periodic checkpoints."""

    def __init__(
        self,
        cfg: TrainingConfig,
        seed: int,
        curriculum: CurriculumManager,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._cfg = cfg
        self._seed = seed
        self._curriculum = curriculum

    def _on_step(self) -> bool:
        if self.n_calls % self._cfg.checkpoint_freq == 0:
            ckpt_path = os.path.join(
                self._cfg.save_dir, f'ppo_step_{self.num_timesteps}')
            self.model.save(ckpt_path)
            save_model_card(
                ckpt_path,
                self._cfg,
                self._seed,
                self._curriculum.current_stage,
                total_timesteps_done=self.num_timesteps,
            )
        return True


# ── Config loading ───────────────────────────────────────────────────────────

# ── Environment factory ─────────────────────────────────────────────────────

def make_env(curriculum: CurriculumManager, reward_weights: RewardWeights, seed: int):
    """Factory function for vectorized environments."""
    def _init():
        from rl_local_planner.gym_env import GazeboExplorerEnv
        return GazeboExplorerEnv(
            curriculum=curriculum,
            reward_weights=reward_weights,
            seed=seed,
        )
    return _init


def make_mujoco_env(mjcf_path: str, curriculum: CurriculumManager,
                    reward_weights: RewardWeights, seed: int):
    """Factory function for MuJoCo vectorized environments."""
    def _init():
        from rl_local_planner.mujoco_env import MuJoCoExplorerEnv
        return MuJoCoExplorerEnv(
            mjcf_path,
            curriculum=curriculum,
            reward_weights=reward_weights,
            seed=seed,
        )
    return _init


def make_point2d_env(curriculum: CurriculumManager,
                     reward_weights: RewardWeights, seed: int):
    """Factory function for lightweight 2D point-mass environments."""
    def _init():
        from rl_local_planner.point2d_env import Point2DExplorerEnv
        return Point2DExplorerEnv(
            curriculum=curriculum,
            reward_weights=reward_weights,
            seed=seed,
        )
    return _init


# ── Main ─────────────────────────────────────────────────────────────────────

@hydra.main(config_path='../conf', config_name='config', version_base=None)
def main(hydra_cfg: DictConfig) -> None:
    # Convert OmegaConf DictConfig → TrainingConfig dataclass (validates types)
    cfg = TrainingConfig(**OmegaConf.to_container(hydra_cfg, resolve=True))

    # ── Run ID ────────────────────────────────────────────────────────────
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]

    # ── Logging ───────────────────────────────────────────────────────────
    log = setup_logging(cfg.tb_log_dir, run_id)
    log.info('Run ID: %s', run_id)
    log.info('sim=%s  seed=%d  num_envs=%d', cfg.sim, cfg.seed, cfg.num_envs)
    log.info('Config: %s', vars(cfg))

    # ── Seeding ───────────────────────────────────────────────────────────
    seed_everything(cfg.seed, log)

    # ── Reward weights ───────────────────────────────────────────────────
    reward_weights = RewardWeights(
        progress=cfg.reward_progress,
        goal_reached=cfg.reward_goal_reached,
        collision=cfg.reward_collision,
        proximity=cfg.reward_proximity,
        smoothness=cfg.reward_smoothness,
        step_cost=cfg.reward_step_cost,
        heading=cfg.reward_heading,
        near_goal=cfg.reward_near_goal,
        near_goal_radius=cfg.reward_near_goal_radius,
        proximity_threshold=cfg.reward_proximity_threshold,
        goal_tolerance=cfg.goal_tolerance,
    )

    # ── Curriculum ───────────────────────────────────────────────────────
    curriculum = CurriculumManager()

    # ── Environment ──────────────────────────────────────────────────────
    num_envs = cfg.num_envs

    if cfg.sim == 'point2d':
        log.info('Point2D backend — pure Python, no external simulator')
        env_fns = [make_point2d_env(curriculum, reward_weights, cfg.seed + i * 1000)
                   for i in range(num_envs)]
        eval_env_fns = [make_point2d_env(curriculum, reward_weights, cfg.seed + 1000)]
    elif cfg.sim == 'mujoco':
        mjcf_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'autonomous_explorer',
            'urdf', 'worlds', 'mujoco_maze.xml')
        mjcf_path = os.path.abspath(mjcf_path)
        log.info('MuJoCo backend — MJCF: %s', mjcf_path)
        env_fns = [make_mujoco_env(mjcf_path, curriculum, reward_weights, cfg.seed + i * 1000)
                   for i in range(num_envs)]
        eval_env_fns = [make_mujoco_env(mjcf_path, curriculum, reward_weights, cfg.seed + 1000)]
    else:
        # Gazebo path — UNCHANGED
        import rclpy
        from rl_local_planner.gym_env import GazeboExplorerEnv  # noqa: F401 (used in make_env)
        if not rclpy.ok():
            rclpy.init()
        env_fns = [make_env(curriculum, reward_weights, cfg.seed + i * 1000)
                   for i in range(num_envs)]
        eval_env_fns = [make_env(curriculum, reward_weights, cfg.seed + 1000)]

    if num_envs > 1:
        log.info('Using SubprocVecEnv with %d environments', num_envs)
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # VecTransposeImage converts (H, W, C) → (C, H, W) for the CNN.
    # Must be applied to BOTH training and eval envs so the feature extractor
    # receives consistently channels-first tensors during training and evaluation.
    env = VecTransposeImage(env)

    if cfg.normalize_reward:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=cfg.gamma)

    # Separate eval env — must mirror training env wrappers so EvalCallback can
    # sync normalisation stats. training=False prevents stats from being updated
    # during evaluation.
    eval_env = DummyVecEnv(eval_env_fns)
    eval_env = VecTransposeImage(eval_env)
    if cfg.normalize_reward:
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True,
                                gamma=cfg.gamma, training=False)

    # ── Policy kwargs ────────────────────────────────────────────────────
    policy_kwargs = {
        'features_extractor_class': RobotFeatureExtractor,
        'features_extractor_kwargs': {'features_dim': cfg.features_dim},
        'net_arch': {
            'pi': cfg.net_arch_pi,
            'vf': cfg.net_arch_vf,
        },
    }

    # ── PPO model ────────────────────────────────────────────────────────
    if cfg.resume_from and os.path.isfile(f'{cfg.resume_from}.zip'):
        # Load from checkpoint
        log.info('Loading model from checkpoint: %s', cfg.resume_from)
        model = PPO.load(cfg.resume_from, env=env)

        # Load VecNormalize stats if available
        vec_norm_path = os.path.join(cfg.save_dir, 'vec_normalize.pkl')
        if isinstance(env, VecNormalize) and os.path.isfile(vec_norm_path):
            log.info('Loading VecNormalize stats from: %s', vec_norm_path)
            env = VecNormalize.load(vec_norm_path, env)
            model.set_env(env)

        log.info('Resumed from checkpoint at step %d', model.num_timesteps)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log.info('Training device: %s', device)
        model = PPO(
            cfg.policy,
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            verbose=1,
            tensorboard_log=cfg.tb_log_dir,
            policy_kwargs=policy_kwargs,
            seed=cfg.seed,
            device=device,
        )

    # ── Callbacks ────────────────────────────────────────────────────────
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.best_model_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.best_model_dir,
        log_path=cfg.tb_log_dir,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        verbose=1,
    )

    callbacks = CallbackList([
        MetricsCallback(),
        CurriculumCallback(curriculum),
        ModelCardCallback(cfg, cfg.seed, curriculum),
        eval_callback,
    ])

    # ── Train ────────────────────────────────────────────────────────────
    log.info('Starting PPO training for %d steps...', cfg.total_timesteps)
    log.info('TensorBoard: tensorboard --logdir %s', cfg.tb_log_dir)

    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        log.info('Training interrupted by user.')

    # ── Save final model ─────────────────────────────────────────────────
    final_path = os.path.join(cfg.save_dir, 'ppo_final')
    model.save(final_path)
    log.info('Final model saved to: %s', final_path)

    save_model_card(
        final_path,
        cfg,
        cfg.seed,
        curriculum.current_stage,
        total_timesteps_done=cfg.total_timesteps,
    )

    # Export resolved config alongside the model for reproducibility
    config_export_path = os.path.join(cfg.save_dir, 'training_config_used.yaml')
    OmegaConf.save(hydra_cfg, config_export_path)
    log.info('Resolved config exported to: %s', config_export_path)

    if isinstance(env, VecNormalize):
        norm_path = os.path.join(cfg.save_dir, 'vec_normalize.pkl')
        env.save(norm_path)
        log.info('VecNormalize stats saved to: %s', norm_path)

    env.close()
    eval_env.close()
    if cfg.sim == 'gazebo':
        import rclpy
        rclpy.shutdown()
    log.info('Training complete.')


if __name__ == '__main__':
    main()
