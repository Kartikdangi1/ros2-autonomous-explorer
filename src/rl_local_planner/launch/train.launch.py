#!/usr/bin/env python3
"""
train.launch.py
===============
Launches the simulation infrastructure needed for RL training:

  T+ 0 s  Gazebo + robot_state_publisher + parameter_bridge
           + sensor_fusion + localization + obstacle_cluster_node
  T+12 s  SLAM Toolbox (mapping)
  T+17 s  Nav2 (planner + controller + BT) — needed for local_costmap

Does NOT launch the NBV goal provider — the training script drives the
robot via the gym env.

After this launch is running, start training in a separate terminal:
    python3 -m rl_local_planner.scripts.train_ppo [--config ...]
    # or:
    python3 src/rl_local_planner/scripts/train_ppo.py [--config ...]
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
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

# ── Package paths ─────────────────────────────────────────────────────────────
EXPLORER_PKG = get_package_share_directory('autonomous_explorer')
RL_PKG       = get_package_share_directory('rl_local_planner')

# ── Config file paths ─────────────────────────────────────────────────────────
TRAINING_BRIDGE = os.path.join(RL_PKG, 'config', 'training_bridge_params.yaml')
FUSION_PARAMS   = os.path.join(EXPLORER_PKG, 'config', 'sensor_fusion_params.yaml')
RVIZ_CONFIG     = os.path.join(EXPLORER_PKG, 'config', 'rviz_config.rviz')
URDF_FILE       = os.path.join(EXPLORER_PKG, 'urdf', 'robot.urdf.xacro')
WORLD_FILE      = os.path.join(EXPLORER_PKG, 'urdf', 'worlds', 'maze_world.sdf')

# ── Sub-launch file paths ────────────────────────────────────────────────────
LOCALIZATION_LAUNCH = os.path.join(EXPLORER_PKG, 'launch', 'localization.launch.py')
MAPPING_LAUNCH      = os.path.join(EXPLORER_PKG, 'launch', 'mapping.launch.py')
NAVIGATION_LAUNCH   = os.path.join(EXPLORER_PKG, 'launch', 'navigation.launch.py')

# ── Spawn position ────────────────────────────────────────────────────────────
SPAWN_X = -11.0
SPAWN_Y = -11.0
SPAWN_Z = 0.15


def _build_world_with_robot(world_file: str = WORLD_FILE) -> str:
    """Convert robot.urdf.xacro → SDF and inject into a world SDF file.

    Replicates the logic from nav2_exploration.launch.py to statically embed
    the robot in the world (avoids MecanumDrive Configure() race condition).

    Args:
        world_file: Absolute path to the .sdf or .world file to use.
    """
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

    with open(world_file, 'r') as f:
        world_content = f.read()

    if '</world>' not in world_content:
        raise RuntimeError(f'No </world> closing tag found in {world_file}')

    combined = world_content.replace(
        '</world>',
        f'\n    <!-- Robot model (statically embedded) -->\n'
        f'    {model_sdf_str}\n\n  </world>')

    world_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.sdf', delete=False, prefix='rl_training_world_')
    world_tmp.write(combined)
    world_tmp.flush()
    world_tmp_path = world_tmp.name
    world_tmp.close()

    debug_path = '/tmp/rl_training_world_debug.sdf'
    shutil.copy(world_tmp_path, debug_path)
    print(f'[train.launch] Combined SDF: {world_tmp_path}')

    return world_tmp_path


def launch_setup(context, *args, **kwargs):
    """OpaqueFunction body — builds the world and returns all nodes."""

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    headless = LaunchConfiguration('headless').perform(context).lower() == 'true'
    world_name = LaunchConfiguration('world').perform(context)

    # Resolve world file — support world_name without extension
    world_file = os.path.join(EXPLORER_PKG, 'urdf', 'worlds', f'{world_name}.sdf')
    if not os.path.isfile(world_file):
        # Try .world extension (e.g. warehouse.world)
        world_file = os.path.join(EXPLORER_PKG, 'urdf', 'worlds', f'{world_name}.world')
    if not os.path.isfile(world_file):
        raise RuntimeError(f'World file not found for world="{world_name}" in '
                           f'{EXPLORER_PKG}/urdf/worlds/')

    world_tmp_path = _build_world_with_robot(world_file)

    # ── Build Gazebo args ─────────────────────────────────────────────────
    # -r   = run immediately (no pause)
    # -s   = server-only (no GUI) when headless
    # --headless-rendering = off-screen rendering for GPU-less CI
    gz_args = f'-r {world_tmp_path}'
    if headless:
        gz_args = f'-s --headless-rendering -r {world_tmp_path}'

    # ── 1. Gazebo simulation ─────────────────────────────────────────────
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ros_gz_sim'), '/launch/gz_sim.launch.py']),
        launch_arguments={'gz_args': gz_args}.items())

    # ── 2. Robot state publisher ─────────────────────────────────────────
    robot_description = xacro.process_file(URDF_FILE).toxml()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description,
            'publish_frequency': 100.0,
        }])

    # ── 3. Gazebo ↔ ROS2 bridge (training config) ───────────────────────
    parameter_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        parameters=[{'config_file': TRAINING_BRIDGE,
                     'use_sim_time': use_sim_time}],
        output='screen')

    # ── 4. Sensor fusion ─────────────────────────────────────────────────
    sensor_fusion = Node(
        package='sensor_fusion',
        executable='sensor_fusion_node',
        name='sensor_fusion',
        parameters=[FUSION_PARAMS, {'use_sim_time': use_sim_time}],
        output='screen')

    # ── 5. Localization (EKF + IMU filter) ───────────────────────────────
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(LOCALIZATION_LAUNCH),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # ── 6. Mapping (SLAM Toolbox) — delayed 12 s ────────────────────────
    mapping = TimerAction(
        period=12.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(MAPPING_LAUNCH),
            launch_arguments={'use_sim_time': use_sim_time}.items())])

    # ── 7. Navigation (Nav2 — needed for local_costmap) — delayed 17 s ──
    navigation = TimerAction(
        period=17.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(NAVIGATION_LAUNCH),
            launch_arguments={'use_sim_time': use_sim_time}.items())])

    # ── 8. Obstacle cluster node ─────────────────────────────────────────
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

    # ── 9. RViz2 (optional) ──────────────────────────────────────────────
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
        navigation,
        obstacle_cluster_node,
        rviz,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use Gazebo simulation time'),
        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Launch RViz2 for debugging'),
        DeclareLaunchArgument(
            'headless', default_value='true',
            description='Run Gazebo headless (no GUI) — required for cloud/CI training'),
        DeclareLaunchArgument(
            'world', default_value='maze_world',
            description='World name to load from autonomous_explorer/urdf/worlds/ '
                        '(without extension). Options: maze_world, warehouse, '
                        'open_field, corridors'),
        OpaqueFunction(function=launch_setup),
    ])
