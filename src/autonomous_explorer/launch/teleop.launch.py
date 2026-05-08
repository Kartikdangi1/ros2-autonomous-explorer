#!/usr/bin/env python3
"""
teleop.launch.py
================
WASD keyboard teleop with live SLAM mapping — no Nav2, no NBV.

Starts:
  - Ignition Gazebo 6       (world + robot statically embedded)
  - ros_gz_bridge            (/cmd_vel → Gazebo, /scan, /odom, /imu …)
  - robot_state_publisher    (URDF → fixed-joint TFs)
  - localization.launch.py   (Madgwick + EKF → odom→base_link TF)
  - mapping.launch.py        (SLAM Toolbox → /map + map→odom TF, delayed 12 s)
  - wasd_teleop_node         (gnome-terminal window, WASD + QE holonomic)
  - rviz2 (optional)         (robot model + scan + live map)

Usage:
  ros2 launch autonomous_explorer teleop.launch.py
  ros2 launch autonomous_explorer teleop.launch.py use_rviz:=true
  ros2 launch autonomous_explorer teleop.launch.py world:=open_field.sdf
  ros2 launch autonomous_explorer teleop.launch.py slam_delay:=15.0

Key bindings (shown in the xterm window):
  W / S       forward / backward
  A / D       strafe left / right  (holonomic mecanum)
  Q / E       rotate left / right
  SPACE       full stop
  + / -       speed up / down
  ] / [       turn speed up / down
  X / Ctrl-C  exit teleop
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

PKG = get_package_share_directory('autonomous_explorer')

BRIDGE_PARAMS       = os.path.join(PKG, 'config', 'robot_params.yaml')
URDF_FILE           = os.path.join(PKG, 'urdf', 'robot.urdf.xacro')
RVIZ_CONFIG         = os.path.join(PKG, 'config', 'teleop_rviz.rviz')
LOCALIZATION_LAUNCH = os.path.join(PKG, 'launch', 'localization.launch.py')
MAPPING_LAUNCH      = os.path.join(PKG, 'launch', 'mapping.launch.py')

# Spawn position (matches nav2_exploration default)
SPAWN_X =  0.0
SPAWN_Y = -6.0
SPAWN_Z =  0.125   # wheel_radius (0.085) + joint_z (0.04)


def _build_world_with_robot(world_file: str) -> str:
    """Convert robot.urdf.xacro → SDF and inject into the world SDF.
    Returns path to the combined temp SDF file.
    Robot is embedded statically to avoid MecanumDrive Configure() race."""
    robot_description = xacro.process_file(URDF_FILE).toxml()

    urdf_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.urdf', delete=False, prefix='robot_teleop_')
    urdf_tmp.write(robot_description)
    urdf_tmp.flush()
    urdf_tmp.close()

    result = subprocess.run(
        ['ign', 'sdf', '-p', urdf_tmp.name],
        capture_output=True, text=True)
    os.unlink(urdf_tmp.name)

    if result.returncode != 0:
        raise RuntimeError(
            f'ign sdf -p failed:\n{result.stdout}\n{result.stderr}')

    root = ET.fromstring(result.stdout.strip())
    model_elem = root.find('model')
    if model_elem is None:
        world_elem = root.find('world')
        if world_elem is not None:
            model_elem = world_elem.find('model')
    if model_elem is None:
        raise RuntimeError('Could not find <model> in ign sdf -p output')

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
        raise RuntimeError(f'No </world> in {world_file}')

    combined = world_content.replace(
        '</world>',
        f'\n    <!-- Teleop robot (statically embedded) -->\n'
        f'    {model_sdf_str}\n\n  </world>')

    world_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.sdf', delete=False, prefix='teleop_world_')
    world_tmp.write(combined)
    world_tmp.flush()
    world_tmp.close()

    debug_path = '/tmp/teleop_world_debug.sdf'
    shutil.copy(world_tmp.name, debug_path)
    print(f'[teleop] World SDF: {world_tmp.name}  (debug: {debug_path})')
    return world_tmp.name


def launch_setup(context, *args, **kwargs):
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz     = LaunchConfiguration('use_rviz')
    world_name   = LaunchConfiguration('world').perform(context)
    slam_delay   = float(LaunchConfiguration('slam_delay').perform(context))

    world_file = os.path.join(PKG, 'urdf', 'worlds', world_name)
    if not os.path.exists(world_file):
        raise FileNotFoundError(f'World file not found: {world_file}')

    world_tmp_path = _build_world_with_robot(world_file)

    # ── Gazebo ────────────────────────────────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ros_gz_sim'), '/launch/gz_sim.launch.py']),
        launch_arguments={'gz_args': f'-r {world_tmp_path}'}.items())

    # ── Robot state publisher (URDF → fixed-joint TFs) ────────────────────────
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

    # ── Gazebo ↔ ROS2 bridge (/cmd_vel, /odom, /scan, /imu …) ───────────────
    parameter_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='parameter_bridge',
        parameters=[{'config_file': BRIDGE_PARAMS,
                     'use_sim_time': use_sim_time}],
        output='screen')

    # ── Localization (Madgwick + EKF → odom→base_link TF) ───────────────────
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(LOCALIZATION_LAUNCH),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    # ── SLAM Toolbox — delayed so EKF TF and /scan are live first ────────────
    mapping = TimerAction(
        period=slam_delay,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(MAPPING_LAUNCH),
            launch_arguments={'use_sim_time': use_sim_time}.items())])

    # ── WASD teleop (opens in a new gnome-terminal window) ───────────────────
    # --wait: gnome-terminal stays alive until the teleop node exits so the
    # launch system can track lifecycle (Ctrl-C in teleop window stops cleanly).
    teleop = Node(
        package='autonomous_explorer',
        executable='wasd_teleop_node.py',
        name='wasd_teleop',
        output='screen',
        prefix='gnome-terminal --wait --',
        parameters=[{'use_sim_time': use_sim_time}])

    # ── RViz (optional) ───────────────────────────────────────────────────────
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
        gazebo,
        robot_state_publisher,
        parameter_bridge,
        localization,
        mapping,
        teleop,
        rviz,
    ]


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),

        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use Gazebo simulation clock'),

        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Launch RViz2 for sensor visualisation'),

        DeclareLaunchArgument(
            'world', default_value='maze_world.sdf',
            description='World SDF filename inside urdf/worlds/ '
                        '(maze_world.sdf | corridors.sdf | open_field.sdf)'),

        DeclareLaunchArgument(
            'slam_delay', default_value='12.0',
            description='Seconds to wait before starting SLAM Toolbox '
                        '(needs Gazebo + EKF TF live first)'),

        OpaqueFunction(function=launch_setup),
    ])
