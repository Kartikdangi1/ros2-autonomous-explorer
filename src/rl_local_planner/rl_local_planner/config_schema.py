"""Hydra structured config for training_config validation.

Replaces the Pydantic BaseModel. Hydra uses this dataclass to:
  - Validate types at load time (wrong type → error at startup)
  - Provide IDE-friendly autocomplete for all config fields
  - Register the schema with ConfigStore so @hydra.main can inject it

Usage (training):
    @hydra.main(config_path="../conf", config_name="config", version_base=None)
    def main(cfg: DictConfig) -> None:
        config = TrainingConfig(**OmegaConf.to_container(cfg, resolve=True))

Usage (legacy, without Hydra):
    from rl_local_planner.config_schema import TrainingConfig
    cfg = TrainingConfig()   # defaults only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class TrainingConfig:
    """Validated training configuration. All fields have sensible defaults."""

    # Runtime (not saved to model card — set via CLI override)
    sim: str = "mujoco"
    seed: int = 42
    num_envs: int = 1
    resume_from: Optional[str] = None

    # PPO hyperparameters
    policy: str = "MultiInputPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 1_500_000

    # Network architecture
    features_dim: int = 128
    net_arch_pi: List[int] = field(default_factory=lambda: [256, 256])
    net_arch_vf: List[int] = field(default_factory=lambda: [256, 256])

    # Normalisation
    normalize_reward: bool = True

    # Evaluation
    eval_freq: int = 25_000
    n_eval_episodes: int = 10

    # Checkpointing
    checkpoint_freq: int = 50_000
    tb_log_dir: str = "./tb_logs/"
    save_dir: str = "./rl_checkpoints/"
    best_model_dir: str = "./rl_best_model/"

    # Reward weights
    reward_progress: float = 5.0
    reward_goal_reached: float = 25.0
    reward_collision: float = -5.0
    reward_proximity: float = -0.5
    reward_smoothness: float = -0.1
    reward_step_cost: float = -0.03
    reward_heading: float = 0.2
    reward_near_goal: float = 5.0
    reward_near_goal_radius: float = 0.8
    reward_proximity_threshold: float = 0.5
    goal_tolerance: float = 0.3

    def __post_init__(self) -> None:
        assert self.goal_tolerance > 0, f"goal_tolerance must be > 0, got {self.goal_tolerance}"
        assert self.total_timesteps > 0, f"total_timesteps must be > 0, got {self.total_timesteps}"
        assert self.learning_rate > 0, f"learning_rate must be > 0, got {self.learning_rate}"
        assert 0 <= self.gamma <= 1, f"gamma must be in [0, 1], got {self.gamma}"
        assert self.sim in ('gazebo', 'mujoco', 'point2d'), \
            f"sim must be gazebo|mujoco|point2d, got {self.sim}"


# Register with Hydra ConfigStore so @hydra.main can inject this schema
cs = ConfigStore.instance()
cs.store(name="config", node=TrainingConfig)
