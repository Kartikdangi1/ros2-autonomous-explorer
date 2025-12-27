#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # Get package directory
    pkg_name = 'my_awesome_robot'
    pkg_share = get_package_share_directory(pkg_name)
    
    # URDF file path
    urdf_file = os.path.join(pkg_share, 'urdf', 'robot.urdf.xacro')
    rviz_config = os.path.join(pkg_share, 'config', 'rviz_config.rviz')
    gz_world_path = os.path.join(pkg_share, 'urdf', 'worlds', 'maze_world.sdf')
    # gz_world_path = 'empty.sdf'
    # Process xacro file
    robot_description_config = xacro.process_file(urdf_file)
    robot_description = {'robot_description': robot_description_config.toxml()}
    dyn_params_file = os.path.join(pkg_share, 'config', 'dynamic_world_params.yaml')
    # Bridge config file
    bridge_params = os.path.join(pkg_share, 'config', 'robot_params.yaml')
    perception_params_file = os.path.join(pkg_share, 'config', 'perception_params.yaml')
    # Robot State Publisher Node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )
    
    # Joint State Publisher GUI (for testing joints)
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )
    
    # RViz2 Node (add a config file if you have one, e.g., rviz_config = os.path.join(pkg_share, 'rviz', 'config.rviz'); then arguments=['-d', rviz_config])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )
    
    # Launch GZ Sim
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch'), '/gz_sim.launch.py']),
        launch_arguments={'gz_args': f'-r {gz_world_path}'}.items()  # -r to run, -v4 for verbose; replace empty.sdf with your world if needed
    )
    
    # Spawn entity
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'my_awesome_robot',
                    '-topic', 'robot_description',
                    '-x', '0', '-y', '0', '-z', '0.1'],
        output='screen'
    )
    
    # Parameter bridge for topics (cmd_vel, odom, camera_info)
    parameter_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
        output='screen'
    )

    # Dynamic Obstacle Spawner
    obstacle_spawner = Node(
        package='my_awesome_robot',
        executable='dynamic_obstacle_spawner.py',
        parameters=[dyn_params_file],
        output='screen',
        name='dynamic_obstacle_spawner'
    )

    # Image bridge for camera/image_raw
    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/camera'],
        output='screen'
    )
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
    # Bridge Gazebo Sim services needed to spawn/move obstacles
    service_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/maze_world/create@ros_gz_interfaces/srv/SpawnEntity',
            '/world/maze_world/remove@ros_gz_interfaces/srv/DeleteEntity',
            '/world/maze_world/set_pose@ros_gz_interfaces/srv/SetEntityPose',
        ],
        output='screen'
    )
    
    # Perception Node
    perception_node = Node(
        package=pkg_name,
        executable='perception_node.py',
        name='perception_node',
        output='screen',
        parameters=[
            perception_params_file if os.path.exists(perception_params_file) else {},
            {
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'debug_mode': LaunchConfiguration('debug_mode')
            }
        ],
        emulate_tty=True
    )
        
    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node,
        gz_sim,
        spawn_entity,
        obstacle_spawner,
        parameter_bridge,
        service_bridge,
        image_bridge,
        # model_arg,
        # confidence_arg,
        # debug_arg,
        # perception_node
    ])