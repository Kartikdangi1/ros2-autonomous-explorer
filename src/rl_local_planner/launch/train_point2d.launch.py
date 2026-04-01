#!/usr/bin/env python3
"""
train_point2d.launch.py
=======================
Launches PPO training using the lightweight Point2D backend.
Use this to rapidly validate the training pipeline before committing
to a full MuJoCo run.

The Point2D environment is a pure-NumPy 2D kinematic simulator that
runs at 5,000-50,000 FPS with no external dependencies. A 750k step
run completes in minutes rather than hours.

Usage:
    ros2 launch rl_local_planner train_point2d.launch.py
    ros2 launch rl_local_planner train_point2d.launch.py num_envs:=8
    ros2 launch rl_local_planner train_point2d.launch.py total_steps:=200000

Arguments:
    num_envs     — parallel environments (default: 4)
    seed         — random seed (default: 42)
    config       — path to training_config.yaml (default: package config)
    tensorboard  — launch TensorBoard alongside training (default: false)
    tb_port      — TensorBoard port (default: 6006)

Validation checklist (watch in TensorBoard before running MuJoCo):
    curriculum/stage        — must NOT jump to 1 immediately; should advance
                              after genuine successes (~5k-20k steps)
    episode/goal_reached    — must rise above 0 within first 10k steps,
                              reach ≥0.40 before Stage 0→1 advance
    episode/goal_distance   — terminal-only; must trend DOWN from ~0.5m
    train/explained_variance — must rise toward 0.4+ within 50k steps

If any of these fail → there is a deeper reward/curriculum bug.
If all pass → proceed to train_mujoco.launch.py for full training.
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
            '--sim', 'point2d',
            '--num-envs', num_envs,
            '--seed', seed,
            '--config', config,
        ],
        name='ppo_point2d_training',
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
            'num_envs', default_value='4',
            description='Number of parallel Point2D environments'),
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
