#!/usr/bin/env python3
"""
localization.launch.py
======================
Launches the localization stack only:
  - imu_filter_madgwick  (/imu + /magnetometer → /imu/data)
  - ekf_filter_node      (/odom + /imu/data → /odometry/filtered + odom→base_link TF)

TF ownership: odom→base_link is published SOLELY by the EKF node at 30 Hz.
  - Gazebo MecanumDrive publish_odom_tf=false  (disabled in URDF)
  - ros_gz_bridge /tf bridge: DISABLED          (commented out in robot_params.yaml)

Can be launched standalone for localization testing:
  ros2 launch my_awesome_robot localization.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = get_package_share_directory('my_awesome_robot')
EKF_PARAMS = os.path.join(PKG, 'config', 'ekf.yaml')


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use Gazebo simulation time')

    use_sim_time = LaunchConfiguration('use_sim_time')

    # ── Madgwick IMU filter — /imu + /magnetometer → /imu/data ───────────────
    madgwick_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='madgwick_filter',
        parameters=[{
            'use_sim_time': use_sim_time,
            'use_mag': True,
            'publish_tf': False,
            'world_frame': 'enu',
            'frequency': 100.0,
            'gain': 0.01,
            'magnetic_declination_radians': 0.0,
            'remove_gravity_vector': False,
        }],
        remappings=[
            ('imu/data_raw', '/imu'),
            ('imu/mag',      '/magnetometer'),
        ],
        output='screen')

    # ── EKF — fuses /odom + /imu/data → /odometry/filtered + odom→base_link TF
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[EKF_PARAMS],
        remappings=[('/odometry/filtered', '/odometry/filtered')],
        output='screen')

    return LaunchDescription([
        use_sim_time_arg,
        madgwick_filter,
        ekf_node,
    ])
