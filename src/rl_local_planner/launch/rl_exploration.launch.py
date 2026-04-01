#!/usr/bin/env python3
"""
rl_exploration.launch.py
========================
Full autonomous exploration launch with an optional RL local planner.

Usage:
  # Default (DWB controller, same as nav2_exploration.launch.py):
  ros2 launch rl_local_planner rl_exploration.launch.py

  # With RL controller replacing DWB:
  ros2 launch rl_local_planner rl_exploration.launch.py use_rl_controller:=true

When use_rl_controller:=true:
  - All infrastructure launches identically (Gazebo, SLAM, EKF, sensors, NBV)
  - Nav2 controller_server's /cmd_vel is remapped to /cmd_vel_dwb (dead topic)
  - RL controller node publishes the real /cmd_vel
  - DWB still runs (BT's FollowPath needs it) but its output is discarded

When use_rl_controller:=false (default):
  - Delegates entirely to nav2_exploration.launch.py — zero difference
"""

import os
import shutil
import subprocess
import tempfile
import xacro
from xml.etree import ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             OpaqueFunction, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

# ── Package paths ─────────────────────────────────────────────────────────────
EXPLORER_PKG = get_package_share_directory('autonomous_explorer')
RL_PKG       = get_package_share_directory('rl_local_planner')

# ── Config files ──────────────────────────────────────────────────────────────
NAV2_PARAMS    = os.path.join(EXPLORER_PKG, 'config', 'nav2_params.yaml')
BT_XML         = os.path.join(EXPLORER_PKG, 'config', 'exploration_bt.xml')
BRIDGE_PARAMS  = os.path.join(EXPLORER_PKG, 'config', 'robot_params.yaml')
FUSION_PARAMS  = os.path.join(EXPLORER_PKG, 'config', 'sensor_fusion_params.yaml')
RVIZ_CONFIG    = os.path.join(EXPLORER_PKG, 'config', 'rviz_config.rviz')
URDF_FILE      = os.path.join(EXPLORER_PKG, 'urdf', 'robot.urdf.xacro')
WORLD_FILE     = os.path.join(EXPLORER_PKG, 'urdf', 'worlds', 'maze_world.sdf')
RL_PARAMS      = os.path.join(RL_PKG, 'config', 'rl_params.yaml')

# ── Sub-launch paths ─────────────────────────────────────────────────────────
LOCALIZATION_LAUNCH = os.path.join(EXPLORER_PKG, 'launch', 'localization.launch.py')
MAPPING_LAUNCH      = os.path.join(EXPLORER_PKG, 'launch', 'mapping.launch.py')
NAV2_EXPLORATION_LAUNCH = os.path.join(EXPLORER_PKG, 'launch', 'nav2_exploration.launch.py')

# ── Spawn position ────────────────────────────────────────────────────────────
SPAWN_X = -11.0
SPAWN_Y = -11.0
SPAWN_Z = 0.15


def _build_world_with_robot() -> str:
    """Convert robot.urdf.xacro → SDF and inject into maze_world.sdf."""
    robot_description = xacro.process_file(URDF_FILE).toxml()

    urdf_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.urdf', delete=False, prefix='robot_')
    urdf_tmp.write(robot_description)
    urdf_tmp.flush()
    urdf_tmp_path = urdf_tmp.name
    urdf_tmp.close()

    result = subprocess.run(
        ['ign', 'sdf', '-p', urdf_tmp_path],
        capture_output=True, text=True)
    os.unlink(urdf_tmp_path)

    if result.returncode != 0:
        raise RuntimeError(
            f'ign sdf -p failed:\nstdout: {result.stdout}\nstderr: {result.stderr}')

    sdf_text = result.stdout.strip()
    root = ET.fromstring(sdf_text)

    model_elem = root.find('model')
    if model_elem is None:
        world_elem = root.find('world')
        if world_elem is not None:
            model_elem = world_elem.find('model')
    if model_elem is None:
        raise RuntimeError(
            f'Could not find <model> in ign sdf -p output:\n{sdf_text[:500]}')

    model_elem.set('name', 'robot')

    pose_elem = model_elem.find('pose')
    if pose_elem is None:
        pose_elem = ET.SubElement(model_elem, 'pose')
    pose_elem.text = f'{SPAWN_X} {SPAWN_Y} {SPAWN_Z} 0 0 0'
    if 'relative_to' in pose_elem.attrib:
        del pose_elem.attrib['relative_to']

    model_sdf_str = ET.tostring(model_elem, encoding='unicode')

    with open(WORLD_FILE, 'r') as f:
        world_content = f.read()

    combined = world_content.replace(
        '</world>',
        f'\n    <!-- Robot model (statically embedded) -->\n'
        f'    {model_sdf_str}\n\n  </world>')

    world_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.sdf', delete=False, prefix='rl_exploration_world_')
    world_tmp.write(combined)
    world_tmp.flush()
    world_tmp_path = world_tmp.name
    world_tmp.close()

    shutil.copy(world_tmp_path, '/tmp/rl_exploration_world_debug.sdf')
    return world_tmp_path


def launch_setup(context, *args, **kwargs):
    """OpaqueFunction body — builds appropriate launch based on RL toggle."""

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    use_rl = LaunchConfiguration('use_rl_controller')
    use_rl_str = use_rl.perform(context)
    rl_enabled = use_rl_str.lower() in ('true', '1', 'yes')

    # ── DEFAULT MODE: delegate entirely to nav2_exploration.launch.py ────
    if not rl_enabled:
        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(NAV2_EXPLORATION_LAUNCH),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'use_rviz': use_rviz,
                }.items()),
        ]

    # ── RL MODE: replicate nav2_exploration with controller remapping ────
    world_tmp_path = _build_world_with_robot()
    robot_description = xacro.process_file(URDF_FILE).toxml()

    # 1. Gazebo
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ros_gz_sim'), '/launch/gz_sim.launch.py']),
        launch_arguments={'gz_args': f'-r {world_tmp_path}'}.items())

    # 2. Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description,
            'publish_frequency': 100.0,
        }])

    # 3. Bridge
    parameter_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        parameters=[{'config_file': BRIDGE_PARAMS,
                     'use_sim_time': use_sim_time}],
        output='screen')

    # 4. Sensor fusion
    sensor_fusion = Node(
        package='sensor_fusion',
        executable='sensor_fusion_node',
        name='sensor_fusion',
        parameters=[FUSION_PARAMS, {'use_sim_time': use_sim_time}],
        output='screen')

    # 5. Localization
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(LOCALIZATION_LAUNCH),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # 6. Mapping — delayed 12 s
    mapping = TimerAction(
        period=12.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(MAPPING_LAUNCH),
            launch_arguments={'use_sim_time': use_sim_time}.items())])

    # 7. Nav2 — delayed 17 s — with controller_server remapped
    #    Controller output goes to /cmd_vel_dwb (dead topic) so RL node
    #    owns /cmd_vel.  Everything else (planner, BT, behaviors) unchanged.
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        parameters=[NAV2_PARAMS],
        output='screen')

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        parameters=[NAV2_PARAMS],
        remappings=[('/cmd_vel', '/cmd_vel_dwb')],  # ← silenced
        arguments=['--ros-args', '--log-level', 'controller_server:=WARN'],
        output='screen')

    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        parameters=[NAV2_PARAMS],
        output='screen')

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        parameters=[NAV2_PARAMS,
                    {'default_bt_xml_filename': BT_XML}],
        output='screen')

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

    nav2_stack = TimerAction(
        period=17.0,
        actions=[
            planner_server,
            controller_server,
            behavior_server,
            bt_navigator,
            lifecycle_manager,
        ])

    # 8. Obstacle cluster node
    obstacle_cluster_node = Node(
        package='autonomous_explorer',
        executable='obstacle_cluster_node.py',
        name='obstacle_cluster_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'obstacle_distance_threshold': 2.0,
            'min_cluster_size': 3,
            'cluster_tolerance': 0.3,
        }],
        remappings=[('/scan', '/fused_scan')],
        output='screen')

    # 9. NBV goal provider — delayed 22 s
    nbv_goal_provider = TimerAction(
        period=22.0,
        actions=[Node(
            package='autonomous_explorer',
            executable='nbv_goal_provider_node.py',
            name='nbv_goal_provider',
            parameters=[{
                'use_sim_time': use_sim_time,
                'map_frame': 'map',
                'base_frame': 'base_link',
                'num_sectors': 72,
                'jump_threshold': 1.5,
                'max_range': 19.0,
                'candidate_offset': 1.0,
                'sample_spacing': 1.0,
                'exploration_radius': 15.0,
                'weight_visibility': 3.0,
                'weight_distance': 1.0,
                'weight_orientation': 0.5,
                'num_rays': 72,
                'goal_tolerance': 0.8,
                'min_visibility_threshold': 0.05,
            }],
            output='screen')])

    # 10. RL controller node — delayed 22 s (after Nav2 is up)
    rl_controller = TimerAction(
        period=22.0,
        actions=[Node(
            package='rl_local_planner',
            executable='rl_controller_node.py',
            name='rl_controller',
            parameters=[RL_PARAMS, {'use_sim_time': use_sim_time}],
            output='screen')])

    # 11. RViz2
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', RVIZ_CONFIG],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
        output='screen')

    return [
        SetParameter(name='use_sim_time', value=use_sim_time),
        gz_sim,
        robot_state_publisher,
        parameter_bridge,
        sensor_fusion,
        localization,
        mapping,
        nav2_stack,
        obstacle_cluster_node,
        nbv_goal_provider,
        rl_controller,
        rviz,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use Gazebo simulation time'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Launch RViz2'),
        DeclareLaunchArgument(
            'use_rl_controller', default_value='false',
            description='Replace DWB with RL local planner'),
        OpaqueFunction(function=launch_setup),
    ])
