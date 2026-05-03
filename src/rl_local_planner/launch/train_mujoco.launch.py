#!/usr/bin/env python3
"""
train_mujoco.launch.py
======================
Launches PPO training using the headless MuJoCo backend.
No ROS2 simulation infrastructure is required — MuJoCo runs entirely
inside the training process.

Usage:
    ros2 launch rl_local_planner train_mujoco.launch.py
    ros2 launch rl_local_planner train_mujoco.launch.py num_envs:=4 seed:=123
    ros2 launch rl_local_planner train_mujoco.launch.py tensorboard:=true

Arguments:
    num_envs     — parallel MuJoCo environments (default: 8)
    seed         — random seed (default: 42)
    config       — path to training_config.yaml (default: package config)
    tensorboard  — launch TensorBoard alongside training (default: false)
    tb_port      — TensorBoard port (default: 6006)

Expected speed: 50-200 FPS × num_envs environments.
750k steps with 8 envs typically completes in a few hours.

Monitor training:
    tensorboard --logdir ./tb_logs/

Healthy training signals to watch:
    curriculum/stage        — advances 0→1 after ~5k-20k steps
    episode/goal_reached    — rises above 0.40 before Stage 0→1 advance
    episode/goal_distance   — trends down from ~0.5m toward 0 (terminal only)
    train/explained_variance — rises toward 0.4-0.6 by 50k steps
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

RL_PKG = get_package_share_directory('rl_local_planner')
DEFAULT_CONFIG = os.path.join(RL_PKG, 'config', 'training_config.yaml')

# Absolute path to the training script (works regardless of CWD)
TRAIN_SCRIPT = os.path.join(RL_PKG, '..', '..', '..', '..', 'src',
                             'rl_local_planner', 'scripts', 'train_ppo.py')
TRAIN_SCRIPT = os.path.abspath(TRAIN_SCRIPT)


def launch_setup(context, *args, **kwargs):
    num_envs = LaunchConfiguration('num_envs').perform(context)
    seed = LaunchConfiguration('seed').perform(context)
    config = LaunchConfiguration('config').perform(context)
    tensorboard = LaunchConfiguration('tensorboard').perform(context).lower() == 'true'
    tb_port = LaunchConfiguration('tb_port').perform(context)

    actions = []

    # ── Training process ──────────────────────────────────────────────────
    actions.append(ExecuteProcess(
        cmd=[
            'python3', TRAIN_SCRIPT,
            '--sim', 'mujoco',
            '--num-envs', num_envs,
            '--seed', seed,
            '--config', config,
        ],
        name='ppo_mujoco_training',
        output='screen',
        emulate_tty=True,
    ))

    # ── TensorBoard (optional) ────────────────────────────────────────────
    if tensorboard:
        actions.append(ExecuteProcess(
            cmd=['tensorboard', '--logdir', './tb_logs/', '--port', tb_port],
            name='tensorboard',
            output='screen',
        ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'num_envs', default_value='8',
            description='Number of parallel MuJoCo environments'),
        DeclareLaunchArgument(
            'seed', default_value='42',
            description='Random seed for reproducibility'),
        DeclareLaunchArgument(
            'config', default_value=DEFAULT_CONFIG,
            description='Path to training_config.yaml'),
        DeclareLaunchArgument(
            'tensorboard', default_value='false',
            description='Launch TensorBoard alongside training'),
        DeclareLaunchArgument(
            'tb_port', default_value='6006',
            description='TensorBoard port'),
        OpaqueFunction(function=launch_setup),
    ])
