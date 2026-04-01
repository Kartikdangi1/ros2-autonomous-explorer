"""Pydantic schema for training_config.yaml validation.

Catches typos and type errors at load time instead of silently falling
back to defaults.  Every field in training_config.yaml has a corresponding
typed attribute here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Validated training configuration with sensible defaults."""

    model_config = {"extra": "forbid"}  # reject unknown keys (catches typos)

    # PPO hyperparameters
    policy: str = "MultiInputPolicy"
    learning_rate: float = Field(3e-4, gt=0)
    n_steps: int = Field(2048, gt=0)
    batch_size: int = Field(64, gt=0)
    n_epochs: int = Field(5, gt=0)
    gamma: float = Field(0.99, ge=0, le=1)
    gae_lambda: float = Field(0.95, ge=0, le=1)
    clip_range: float = Field(0.2, gt=0, le=1)
    ent_coef: float = Field(0.02, ge=0)
    total_timesteps: int = Field(500_000, gt=0)

    # Network architecture
    features_dim: int = Field(128, gt=0)
    net_arch_pi: list[int] = Field(default_factory=lambda: [256, 256])
    net_arch_vf: list[int] = Field(default_factory=lambda: [256, 256])

    # Normalisation
    normalize_reward: bool = True

    # Evaluation
    eval_freq: int = Field(25_000, gt=0)
    n_eval_episodes: int = Field(10, gt=0)

    # Checkpointing
    checkpoint_freq: int = Field(50_000, gt=0)
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
    goal_tolerance: float = 0.6
