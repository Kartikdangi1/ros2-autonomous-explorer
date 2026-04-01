#!/usr/bin/env python3
"""Export the best SB3 PPO checkpoint to ONNX for inference.

Usage:
    python3 export_onnx.py [--checkpoint path/to/best_model.zip] [--output model.onnx]

The exported ONNX model takes the Dict observation as separate named
inputs (costmap, scan, goal_vector, velocity) and outputs the action
mean (deterministic policy).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from stable_baselines3 import PPO

from rl_local_planner.obs_builder import COSTMAP_OBS_SIZE, SCAN_DIM


def export_to_onnx(checkpoint_path: str, output_path: str) -> None:
    """Load SB3 model and export the policy network to ONNX."""

    print(f'Loading model from: {checkpoint_path}')
    model = PPO.load(checkpoint_path, device='cpu')

    policy = model.policy
    policy.eval()

    # Create dummy observations matching the Dict observation space
    dummy_obs = {
        'costmap': torch.zeros(1, COSTMAP_OBS_SIZE, COSTMAP_OBS_SIZE, 1,
                               dtype=torch.float32),
        'scan': torch.zeros(1, SCAN_DIM, dtype=torch.float32),
        'goal_vector': torch.zeros(1, 2, dtype=torch.float32),
        'velocity': torch.zeros(1, 3, dtype=torch.float32),
    }

    # The policy's forward method returns actions, values, log_probs.
    # For inference we only need the deterministic action (action_net output).
    # We export a wrapper that returns just the mean action.

    class PolicyWrapper(torch.nn.Module):
        """Wraps the SB3 policy to output deterministic action only."""

        def __init__(self, sb3_policy):
            super().__init__()
            self.features_extractor = sb3_policy.features_extractor
            self.mlp_extractor = sb3_policy.mlp_extractor
            self.action_net = sb3_policy.action_net

        def forward(
            self,
            costmap: torch.Tensor,
            scan: torch.Tensor,
            goal_vector: torch.Tensor,
            velocity: torch.Tensor,
        ) -> torch.Tensor:
            obs = {
                'costmap': costmap,
                'scan': scan,
                'goal_vector': goal_vector,
                'velocity': velocity,
            }
            features = self.features_extractor(obs)
            latent_pi, _ = self.mlp_extractor(features)
            return self.action_net(latent_pi)

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    print(f'Exporting to ONNX: {output_path}')
    torch.onnx.export(
        wrapper,
        (
            dummy_obs['costmap'],
            dummy_obs['scan'],
            dummy_obs['goal_vector'],
            dummy_obs['velocity'],
        ),
        output_path,
        input_names=['costmap', 'scan', 'goal_vector', 'velocity'],
        output_names=['action'],
        dynamic_axes={
            'costmap': {0: 'batch'},
            'scan': {0: 'batch'},
            'goal_vector': {0: 'batch'},
            'velocity': {0: 'batch'},
            'action': {0: 'batch'},
        },
        opset_version=14,
    )

    # Verify the exported model
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    result = sess.run(None, {
        'costmap': dummy_obs['costmap'].numpy(),
        'scan': dummy_obs['scan'].numpy(),
        'goal_vector': dummy_obs['goal_vector'].numpy(),
        'velocity': dummy_obs['velocity'].numpy(),
    })
    print(f'Verification passed. Output shape: {result[0].shape}')
    print(f'ONNX model saved to: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Export SB3 PPO to ONNX')
    parser.add_argument(
        '--checkpoint', type=str,
        default='./rl_best_model/best_model.zip',
        help='Path to SB3 model checkpoint (.zip)')
    parser.add_argument(
        '--output', type=str,
        default='./models/explorer_ppo.onnx',
        help='Output ONNX file path')
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f'Error: checkpoint not found: {args.checkpoint}')
        print('Train a model first with train_ppo.py')
        return

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    export_to_onnx(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
