#!/usr/bin/env python3
"""
rl_exploration.launch.py — compatibility shim
=============================================
Deprecated. Use nav2_exploration.launch.py with controller:=rl instead:

  ros2 launch autonomous_explorer nav2_exploration.launch.py controller:=rl use_rviz:=true

This file is kept for backward compatibility only.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

EXPLORER_PKG = get_package_share_directory('autonomous_explorer')
NAV2_EXPLORATION_LAUNCH = os.path.join(
    EXPLORER_PKG, 'launch', 'nav2_exploration.launch.py')


def generate_launch_description():

    use_rl_arg = DeclareLaunchArgument(
        'use_rl_controller', default_value='true',
        description='Deprecated arg — kept for compatibility. '
                    'Use nav2_exploration.launch.py controller:=rl directly.')

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz2')

    # Map legacy use_rl_controller:=true/false → controller:=rl/dwb
    # LaunchConfiguration('use_rl_controller') == 'true' → 'rl', else 'dwb'
    # We always pass controller:=rl since the purpose of this file is RL mode.
    return LaunchDescription([
        use_rl_arg,
        use_rviz_arg,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(NAV2_EXPLORATION_LAUNCH),
            launch_arguments={
                'controller': 'rl',
                'use_rviz': LaunchConfiguration('use_rviz'),
            }.items()),
    ])
