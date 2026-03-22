#!/usr/bin/env python3
"""
nav2_exploration.launch.py
==========================
Top-level orchestration launch.  Starts all subsystems in dependency order:

  T+ 0 s  Simulation infrastructure
            ├─ Ignition Gazebo 6      (world + robot model, statically embedded)
            ├─ robot_state_publisher  (URDF → fixed-joint TFs at 100 Hz)
            ├─ parameter_bridge       (Gazebo ↔ ROS 2 topic bridging)
            ├─ sensor_fusion          (/scan + /radar + /depth → /fused_scan)
            ├─ localization.launch.py (Madgwick filter + EKF → odom→base_link TF)
            ├─ frontend_node          (obstacle clustering → /detected_obstacles)
            └─ RViz2                  (optional)

  T+12 s  mapping.launch.py   (SLAM Toolbox → /map + map→odom TF)
            Waits for: EKF TF live, /scan publishing from Gazebo

  T+17 s  navigation.launch.py (Nav2 planner + controller + BT navigator)
            Waits for: /map published by SLAM Toolbox

  T+22 s  nbv_goal_provider_node  (NBV exploration mission controller)
            Waits for: Nav2 navigate_to_pose action server ready

TF tree (authoritative, no duplicate publishers):
  map ──(slam_toolbox)──> odom ──(EKF)──> base_link ──(URDF)──> lidar_link
                                                               ──> camera_link
                                                               ──> ...

Static-spawn approach (fixes MecanumDrive Configure() race):
  Robot is embedded directly in maze_world.sdf before Gazebo starts.
  See _build_world_with_robot() for details.

Individual subsystems can be launched standalone for debugging:
  ros2 launch my_awesome_robot localization.launch.py
  ros2 launch my_awesome_robot mapping.launch.py
  ros2 launch my_awesome_robot navigation.launch.py
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
                             OpaqueFunction, SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

# ── Package paths ─────────────────────────────────────────────────────────────
PKG     = get_package_share_directory('my_awesome_robot')

# ── Config file paths ─────────────────────────────────────────────────────────
BRIDGE_PARAMS  = os.path.join(PKG, 'config', 'robot_params.yaml')
FUSION_PARAMS  = os.path.join(PKG, 'config', 'sensor_fusion_params.yaml')
RVIZ_CONFIG    = os.path.join(PKG, 'config', 'rviz_config.rviz')
URDF_FILE      = os.path.join(PKG, 'urdf', 'robot.urdf.xacro')
WORLD_FILE     = os.path.join(PKG, 'urdf', 'worlds', 'maze_world.sdf')

# ── Sub-launch file paths ─────────────────────────────────────────────────────
LOCALIZATION_LAUNCH = os.path.join(PKG, 'launch', 'localization.launch.py')
MAPPING_LAUNCH      = os.path.join(PKG, 'launch', 'mapping.launch.py')
NAVIGATION_LAUNCH   = os.path.join(PKG, 'launch', 'navigation.launch.py')

# ── Spawn position ────────────────────────────────────────────────────────────
# World (0, -3): open central area with ~3 m clearance to all nearest walls.
# (Old SW corner at (-11,-11) left only 1.5 m to perimeter walls, causing the
#  first NBV goals to hug the west wall and the global planner to route into it.)
SPAWN_X = -11.0
SPAWN_Y = -11.0
SPAWN_Z = 0.15


def _build_world_with_robot() -> str:
    """
    Convert robot.urdf.xacro → SDF and inject the <model> element directly
    into a copy of maze_world.sdf.  Returns the path to the temporary combined
    SDF file.

    Why static embedding instead of ros_gz_sim create:
      In Ignition Fortress, UserCommands::Implementation::ProcessEntityCreate()
      commits the new model entity to the ECM in the *same* simulation step that
      the model's plugins are loaded.  MecanumDrive::Configure() therefore runs
      before the joint child-entities are available, so
        _ecm->Component<components::Joint>(jointEntity)
      returns nullptr for every wheel joint and the plugin marks itself as
      uninitialised.  PreUpdate() bails out on the nullptr check and the robot
      never receives velocity commands.  Models that are defined statically in
      the world SDF are committed to the ECM during world loading (before the
      first simulation step), so Configure() sees all child entities.
    """
    # 1. xacro → URDF string
    robot_description = xacro.process_file(URDF_FILE).toxml()

    # 2. Write URDF to a named temp file (ign sdf -p needs a real path)
    urdf_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.urdf', delete=False, prefix='robot_')
    urdf_tmp.write(robot_description)
    urdf_tmp.flush()
    urdf_tmp_path = urdf_tmp.name
    urdf_tmp.close()

    # 3. Convert URDF → SDF
    result = subprocess.run(
        ['ign', 'sdf', '-p', urdf_tmp_path],
        capture_output=True, text=True)
    os.unlink(urdf_tmp_path)

    if result.returncode != 0:
        raise RuntimeError(
            f'ign sdf -p failed:\nstdout: {result.stdout}\nstderr: {result.stderr}')

    # 4. Parse SDF output: <sdf version="..."><model name="...">...</model></sdf>
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

    # 5. Force the model name to "robot"
    model_elem.set('name', 'robot')

    # 6. Set spawn pose
    pose_elem = model_elem.find('pose')
    if pose_elem is None:
        pose_elem = ET.SubElement(model_elem, 'pose')
    pose_elem.text = f'{SPAWN_X} {SPAWN_Y} {SPAWN_Z} 0 0 0'
    if 'relative_to' in pose_elem.attrib:
        del pose_elem.attrib['relative_to']

    # 7. Serialise the model element
    model_sdf_str = ET.tostring(model_elem, encoding='unicode')

    # 8. Read the base world SDF and inject the robot before </world>
    with open(WORLD_FILE, 'r') as f:
        world_content = f.read()

    if '</world>' not in world_content:
        raise RuntimeError(f'No </world> closing tag found in {WORLD_FILE}')

    combined = world_content.replace(
        '</world>',
        f'\n    <!-- Robot model (statically embedded to avoid MecanumDrive'
        f' Configure() race condition) -->\n'
        f'    {model_sdf_str}\n\n  </world>')

    # 9. Write combined SDF to a temp file
    world_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.sdf', delete=False, prefix='maze_with_robot_')
    world_tmp.write(combined)
    world_tmp.flush()
    world_tmp_path = world_tmp.name
    world_tmp.close()

    # 10. Copy to a fixed path for post-launch inspection
    debug_path = '/tmp/combined_maze_debug.sdf'
    shutil.copy(world_tmp_path, debug_path)
    print(f'[nav2_exploration] Combined SDF written to: {world_tmp_path}')
    print(f'[nav2_exploration] Debug copy at:           {debug_path}')

    return world_tmp_path


def launch_setup(context, *args, **kwargs):
    """OpaqueFunction body — builds the combined world SDF then returns all nodes."""

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz     = LaunchConfiguration('use_rviz')

    # ── Generate combined world (robot embedded statically) ───────────────────
    world_tmp_path = _build_world_with_robot()

    # ── 1. Gazebo simulation ──────────────────────────────────────────────────
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ros_gz_sim'), '/launch/gz_sim.launch.py']),
        launch_arguments={'gz_args': f'-r {world_tmp_path}'}.items())

    # ── 2. Robot model: URDF → TF for fixed joints ────────────────────────────
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

    # ── 3. Gazebo ↔ ROS2 bridge ────────────────────────────────────────────────
    parameter_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        parameters=[{'config_file': BRIDGE_PARAMS,
                     'use_sim_time': use_sim_time}],
        output='screen')

    # ── 4. Sensor fusion — /scan + /radar/scan + depth → /fused_scan ─────────
    sensor_fusion = Node(
        package='sensor_fusion_preprocess',
        executable='sensor_fusion_preprocess_node',
        name='sensor_fusion_preprocess',
        parameters=[FUSION_PARAMS, {'use_sim_time': use_sim_time}],
        output='screen')

    # ── 5. Localization — Madgwick filter + EKF (odom→base_link TF) ──────────
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(LOCALIZATION_LAUNCH),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # ── 6. Mapping — SLAM Toolbox (map + map→odom TF) — delayed 12 s ─────────
    # Waits for: Gazebo physics started, /scan publishing, EKF TF live.
    # SLAM uses /scan (raw LiDAR, BEST_EFFORT QoS) for reliable scan delivery.
    mapping = TimerAction(
        period=12.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(MAPPING_LAUNCH),
            launch_arguments={'use_sim_time': use_sim_time}.items())])

    # ── 7. Navigation — Nav2 planner + controller + BT — delayed 17 s ────────
    # Waits for: /map published by SLAM Toolbox (needs ~3-4 s after SLAM start)
    navigation = TimerAction(
        period=17.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(NAVIGATION_LAUNCH),
            launch_arguments={'use_sim_time': use_sim_time}.items())])

    # ── 8. Obstacle detection (frontend) ──────────────────────────────────────
    frontend_node = Node(
        package='my_awesome_robot',
        executable='frontend_node.py',
        name='frontend_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'obstacle_distance_threshold': 2.0,
            'min_cluster_size': 3,
            'cluster_tolerance': 0.3,
        }],
        remappings=[
            ('/scan', '/fused_scan'),
            ('/detected_obstacles', '/detected_obstacles'),
        ],
        output='screen')

    # ── 9. NBV goal provider — mission controller — delayed 22 s ──────────────
    # Waits for: Nav2 navigate_to_pose action server active (lifecycle complete)
    nbv_goal_provider = TimerAction(
        period=22.0,
        actions=[Node(
            package='my_awesome_robot',
            executable='nbv_goal_provider_node.py',
            name='nbv_goal_provider',
            parameters=[{
                'use_sim_time': use_sim_time,
                'map_frame': 'map',
                'base_frame': 'base_link',
                'num_sectors': 72,
                'jump_threshold': 1.5,
                'max_range': 19.0,
                'candidate_offset': 1.0,   # was 0.5 — keeps goals ≥1 m from walls
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

    # ── 10. RViz2 ─────────────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', RVIZ_CONFIG],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
        output='screen')

    return [
        # Propagate use_sim_time to all nodes (including those in sub-launches)
        SetParameter(name='use_sim_time', value=use_sim_time),

        # Simulation infrastructure
        gz_sim,
        robot_state_publisher,
        parameter_bridge,

        # Perception + Localization (T+0)
        sensor_fusion,
        localization,

        # Mapping (T+12, after EKF is publishing TF)
        mapping,

        # Navigation (T+17, after SLAM has published first /map)
        navigation,

        # Obstacle detection
        frontend_node,

        # Mission controller (T+22, after Nav2 action server is active)
        nbv_goal_provider,

        # Visualisation
        rviz,
    ]


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use Gazebo simulation time')

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz2')

    return LaunchDescription([
        use_sim_time_arg,
        use_rviz_arg,
        OpaqueFunction(function=launch_setup),
    ])
