#!/usr/bin/env python3
"""
navigation.launch.py
====================
Launches the Nav2 navigation stack only:
  - planner_server    (SmacPlanner2D — global A* path planning)
  - controller_server (DWB holonomic — local path following)
  - behavior_server   (spin / backup / wait recovery)
  - bt_navigator      (NavigateToPose via behavior tree)
  - lifecycle_manager (autostart-activates all servers above)

Prerequisites before starting:
  - /map must be publishing     (SLAM Toolbox from mapping.launch.py)
  - map→odom TF must be live    (SLAM Toolbox)
  - odom→base_link TF must live (EKF from localization.launch.py)
  - /fused_scan must publish    (sensor_fusion for obstacle layers)

Startup delay is managed by nav2_exploration.launch.py (TimerAction ≥ 17 s).

Can be launched standalone (after localization + mapping are running):
  ros2 launch my_awesome_robot navigation.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = get_package_share_directory('my_awesome_robot')
NAV2_PARAMS = os.path.join(PKG, 'config', 'nav2_params.yaml')
BT_XML      = os.path.join(PKG, 'config', 'exploration_bt.xml')


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use Gazebo simulation time')

    use_sim_time = LaunchConfiguration('use_sim_time')

    # ── Global path planner ───────────────────────────────────────────────────
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        parameters=[NAV2_PARAMS],
        output='screen')

    # ── Local path controller ─────────────────────────────────────────────────
    # --log-level WARN suppresses the ~1 Hz "Passing new path to controller"
    # INFO spam that appears during every active navigation goal.
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        parameters=[NAV2_PARAMS],
        remappings=[('/cmd_vel', '/cmd_vel')],
        arguments=['--ros-args', '--log-level', 'controller_server:=WARN'],
        output='screen')

    # ── Recovery behaviors ────────────────────────────────────────────────────
    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        parameters=[NAV2_PARAMS],
        output='screen')

    # ── BT Navigator — NavigateToPose via exploration_bt.xml ─────────────────
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        parameters=[NAV2_PARAMS,
                    {'default_bt_xml_filename': BT_XML}],
        output='screen')

    # ── Lifecycle manager — activates planner/controller/behavior/navigator ──
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': [
                'planner_server',
                'controller_server',
                'behavior_server',
                'bt_navigator',
            ],
        }],
        output='screen')

    return LaunchDescription([
        use_sim_time_arg,
        planner_server,
        controller_server,
        behavior_server,
        bt_navigator,
        lifecycle_manager,
    ])
