#!/usr/bin/env python3
"""
mapping.launch.py
=================
Launches the mapping stack only:
  - async_slam_toolbox_node  (/scan → /map + map→odom TF)

Scan source: /scan (raw LiDAR, BEST_EFFORT QoS from Gazebo bridge).
  This matches slam_toolbox's sensorDataQoS (BEST_EFFORT) subscription.
  Nav2 obstacle_layer and NBV goal provider use /fused_scan separately.

TF ownership: map→odom is published SOLELY by slam_toolbox (dynamic,
  loop-closure corrected).

Prerequisites before starting this node:
  - /scan must be publishing  (Gazebo bridge + LiDAR sensor active)
  - odom→base_link TF must be live  (EKF from localization.launch.py)
  - base_link→lidar_link TF must be live  (robot_state_publisher from URDF)

Startup delay is managed by nav2_exploration.launch.py (TimerAction ≥ 12 s).

Can be launched standalone for mapping tests (after localization is running):
  ros2 launch autonomous_explorer mapping.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

PKG = get_package_share_directory('autonomous_explorer')
SLAM_PARAMS = os.path.join(PKG, 'config', 'slam_toolbox_params.yaml')


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use Gazebo simulation time')

    # ── SLAM Toolbox async — /scan → /map (OccupancyGrid) + map→odom TF ──────
    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[SLAM_PARAMS],
        output='screen')

    return LaunchDescription([
        use_sim_time_arg,
        slam_toolbox,
    ])
