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
            ├─ obstacle_cluster_node          (obstacle clustering → /detected_obstacles)
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
  ros2 launch autonomous_explorer localization.launch.py
  ros2 launch autonomous_explorer mapping.launch.py
  ros2 launch autonomous_explorer navigation.launch.py
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
PKG     = get_package_share_directory('autonomous_explorer')
RL_PKG  = get_package_share_directory('rl_local_planner')

# ── Config file paths ─────────────────────────────────────────────────────────
BRIDGE_PARAMS  = os.path.join(PKG, 'config', 'robot_params.yaml')
FUSION_PARAMS  = os.path.join(PKG, 'config', 'sensor_fusion_params.yaml')
RVIZ_CONFIG    = os.path.join(PKG, 'config', 'rviz_config.rviz')
URDF_FILE      = os.path.join(PKG, 'urdf', 'robot.urdf.xacro')
WORLD_FILE     = os.path.join(PKG, 'urdf', 'worlds', 'maze_world.sdf')
RL_PARAMS      = os.path.join(RL_PKG, 'config', 'rl_params.yaml')

# ── Sub-launch file paths ─────────────────────────────────────────────────────
LOCALIZATION_LAUNCH = os.path.join(PKG, 'launch', 'localization.launch.py')
MAPPING_LAUNCH      = os.path.join(PKG, 'launch', 'mapping.launch.py')
NAVIGATION_LAUNCH   = os.path.join(PKG, 'launch', 'navigation.launch.py')

# ── Spawn position ────────────────────────────────────────────────────────────
# South room: open area below wall_h_south_corridor (y=-5), well clear of all walls.
# (-11,-11) had ramp_south at (-9,-9,tilted) and start_marker at (-10,-10)
# right next to spawn, immediately distorting SLAM and physically wedging the robot.
SPAWN_X = -3.0
SPAWN_Y = -8.0
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

    slam_delay  = float(LaunchConfiguration('slam_delay').perform(context))
    nav2_delay  = float(LaunchConfiguration('nav2_delay').perform(context))
    nbv_delay   = float(LaunchConfiguration('nbv_delay').perform(context))
    controller  = LaunchConfiguration('controller').perform(context)

    # When RL controller is active, remap DWB output to a dead topic so it
    # doesn't fight with the RL node over /cmd_vel.
    cmd_vel_topic = '/cmd_vel_dwb' if controller == 'rl' else '/cmd_vel'

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
        package='sensor_fusion',
        executable='sensor_fusion_node',
        name='sensor_fusion',
        parameters=[FUSION_PARAMS, {'use_sim_time': use_sim_time}],
        output='screen')

    # ── 5. Localization — Madgwick filter + EKF (odom→base_link TF) ──────────
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(LOCALIZATION_LAUNCH),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # ── 6. Mapping — SLAM Toolbox (map + map→odom TF) — delayed slam_delay s ──
    # Waits for: Gazebo physics started, /scan publishing, EKF TF live.
    # SLAM uses /scan (raw LiDAR, BEST_EFFORT QoS) for reliable scan delivery.
    mapping = TimerAction(
        period=slam_delay,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(MAPPING_LAUNCH),
            launch_arguments={'use_sim_time': use_sim_time}.items())])

    # ── 7. Navigation — Nav2 planner + controller + BT — delayed nav2_delay s ─
    # Waits for: /map published by SLAM Toolbox (needs ~3-4 s after SLAM start)
    # cmd_vel_topic is /cmd_vel (DWB mode) or /cmd_vel_dwb (RL mode, dead topic)
    navigation = TimerAction(
        period=nav2_delay,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(NAVIGATION_LAUNCH),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'cmd_vel_topic': cmd_vel_topic,
            }.items())])

    # ── 8. Obstacle detection (frontend) ──────────────────────────────────────
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
        remappings=[
            ('/scan', '/fused_scan'),
            ('/detected_obstacles', '/detected_obstacles'),
        ],
        output='screen')

    # ── 9. NBV goal provider — mission controller — delayed nbv_delay s ─────────
    # Waits for: Nav2 navigate_to_pose action server active (lifecycle complete)
    nbv_goal_provider = TimerAction(
        period=nbv_delay,
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
                'exploration_radius': 10.0,
                'weight_visibility': 3.0,
                'weight_distance': 1.0,
                'weight_orientation': 0.5,
                'num_rays': 72,
                'min_visibility_threshold': 0.045,
            }],
            output='screen')])

    # ── 10. Path speed limiter — curvature-based DWB speed control — delayed nbv_delay s ──
    # Waits for: Nav2 action server active (same gate as NBV)
    path_speed_limiter = TimerAction(
        period=nbv_delay,
        actions=[Node(
            package='autonomous_explorer',
            executable='path_speed_limiter_node.py',
            name='path_speed_limiter',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen')])

    # ── 11. RL controller (only when controller:=rl) ──────────────────────────
    # Reads the global /plan, runs ONNX inference at 10 Hz, publishes /cmd_vel.
    # DWB still runs but its output is silenced (/cmd_vel_dwb dead topic).
    rl_controller = None
    if controller == 'rl':
        rl_controller = TimerAction(
            period=nbv_delay,
            actions=[Node(
                package='rl_local_planner',
                executable='rl_controller_node.py',
                name='rl_controller',
                parameters=[RL_PARAMS, {'use_sim_time': use_sim_time}],
                output='screen')])

    # ── 12. RViz2 ────────────────────────────────────────────────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', RVIZ_CONFIG],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
        output='screen')

    actions = [
        # Propagate use_sim_time to all nodes (including those in sub-launches)
        SetParameter(name='use_sim_time', value=use_sim_time),

        # Simulation infrastructure
        gz_sim,
        robot_state_publisher,
        parameter_bridge,

        # Perception + Localization (T+0)
        sensor_fusion,
        localization,

        # Mapping (T+slam_delay, after EKF is publishing TF)
        mapping,

        # Navigation (T+nav2_delay, after SLAM has published first /map)
        navigation,

        # Obstacle detection
        obstacle_cluster_node,

        # Mission controller (T+nbv_delay, after Nav2 action server is active)
        nbv_goal_provider,

        # Speed limiter (T+nbv_delay, curvature-based DWB velocity scaling)
        path_speed_limiter,

        # Visualisation
        rviz,
    ]

    if rl_controller is not None:
        actions.append(rl_controller)

    return actions


def generate_launch_description():

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use Gazebo simulation time')

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz2')

    slam_delay_arg = DeclareLaunchArgument(
        'slam_delay', default_value='12.0',
        description='Seconds to wait before starting SLAM Toolbox (after EKF + /scan ready)')

    nav2_delay_arg = DeclareLaunchArgument(
        'nav2_delay', default_value='17.0',
        description='Seconds to wait before starting Nav2 (after SLAM publishes first /map)')

    nbv_delay_arg = DeclareLaunchArgument(
        'nbv_delay', default_value='22.0',
        description='Seconds to wait before starting NBV + speed limiter (after Nav2 action server ready)')

    controller_arg = DeclareLaunchArgument(
        'controller', default_value='dwb',
        description='Local planner backend: dwb (classical DWB) or rl (PPO ONNX policy). '
                    'rl mode: DWB output is silenced, rl_controller_node owns /cmd_vel.')

    return LaunchDescription([
        use_sim_time_arg,
        use_rviz_arg,
        slam_delay_arg,
        nav2_delay_arg,
        nbv_delay_arg,
        controller_arg,
        OpaqueFunction(function=launch_setup),
    ])
