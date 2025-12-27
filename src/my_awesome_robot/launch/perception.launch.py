#!/usr/bin/env python3
"""
Launch file for YOLOv8 perception system

Starts the perception node with parameters loaded from YAML config
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    pkg_name = 'my_awesome_robot'
    pkg_share = get_package_share_directory(pkg_name)
    
    # Config file path
    perception_config = os.path.join(pkg_share, 'config', 'perception_params.yaml')
    
    # Declare launch arguments
    model_arg = DeclareLaunchArgument(
        'model_path',
        default_value='yolov8n.pt',
        description='Path to YOLOv8 model file (e.g., yolov8n.pt, yolov8s.pt)'
    )
    
    confidence_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Minimum confidence threshold for detections'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug logging'
    )
    
    # Perception Node
    perception_node = Node(
        package=pkg_name,
        executable='perception_node.py',
        name='perception_node',
        output='screen',
        parameters=[
            perception_config if os.path.exists(perception_config) else {},
            {
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'debug_mode': LaunchConfiguration('debug_mode')
            }
        ],
        emulate_tty=True
    )
    
    return LaunchDescription([
        model_arg,
        confidence_arg,
        debug_arg,
        perception_node
    ])
