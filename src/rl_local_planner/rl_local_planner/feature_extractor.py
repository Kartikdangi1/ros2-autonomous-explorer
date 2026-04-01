"""Custom multi-input feature extractor for SB3 PPO.

Architecture:
  costmap (84×84×1) → 3-layer CNN → 128-d
  scan    (360,)    → 2-layer MLP → 64-d
  goal    (2,)      ─┐
  velocity(3,)      ─┤ concat → 5-d
                     │
  All branches concatenated → 197-d → Linear → features_dim
"""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RobotFeatureExtractor(BaseFeaturesExtractor):
    """Processes the Dict observation with separate branches per modality."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        # BaseFeaturesExtractor.__init__ stores self.features_dim
        super().__init__(observation_space, features_dim)

        # ── Costmap branch: 3-layer CNN ──────────────────────────────────
        # SB3's VecTransposeImage converts gym (H, W, C) → (C, H, W) before
        # the obs reaches us, so observation_space is already channels-first.
        costmap_shape = observation_space['costmap'].shape  # (1, 84, 84) after VecTransposeImage
        n_channels = costmap_shape[0]

        self.costmap_cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, *costmap_shape)  # (1, C, H, W)
            cnn_out_dim = self.costmap_cnn(dummy).shape[1]

        self.costmap_linear = nn.Sequential(
            nn.Linear(cnn_out_dim, 128),
            nn.ReLU(),
        )

        # ── Scan branch: 2-layer MLP ────────────────────────────────────
        scan_dim = observation_space['scan'].shape[0]  # 360
        self.scan_mlp = nn.Sequential(
            nn.Linear(scan_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # ── Goal + velocity pass-through ─────────────────────────────────
        goal_dim = observation_space['goal_vector'].shape[0]  # 2
        vel_dim = observation_space['velocity'].shape[0]       # 3
        vector_dim = goal_dim + vel_dim                        # 5

        # ── Final projection ─────────────────────────────────────────────
        combined_dim = 128 + 64 + vector_dim  # 197
        self.final_linear = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # Costmap arrives as (B, C, H, W) — already transposed by VecTransposeImage
        costmap = observations['costmap'].float() / 255.0
        costmap_features = self.costmap_linear(self.costmap_cnn(costmap))

        # Scan
        scan_features = self.scan_mlp(observations['scan'])

        # Goal + velocity
        vectors = torch.cat([
            observations['goal_vector'],
            observations['velocity'],
        ], dim=1)

        # Concatenate all branches
        combined = torch.cat([costmap_features, scan_features, vectors], dim=1)
        return self.final_linear(combined)
