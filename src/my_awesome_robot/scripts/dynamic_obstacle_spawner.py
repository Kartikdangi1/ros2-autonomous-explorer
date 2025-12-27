#!/usr/bin/env python3
"""
Dynamic Obstacle Spawner for ROS 2 + Gazebo Sim (Ignition / gz-sim)

Spawns simple box models that represent COCO-like classes:
- person (red)
- dog (orange)
- cat (yellow)

Moves them in circular trajectories using:
  /world/<world>/set_pose (bridged as ros_gz_interfaces/srv/SetEntityPose)

Requires Gazebo Sim UserCommands system (create/remove/set_pose services). [web:208]
"""

import math
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

from ros_gz_interfaces.msg import Entity, EntityFactory
from ros_gz_interfaces.srv import SpawnEntity, DeleteEntity, SetEntityPose


class DynamicObstacleSpawner(Node):
    def __init__(self):
        super().__init__("dynamic_obstacle_spawner")

        # ---- Parameters
        self.declare_parameter("world_name", "maze_world")
        self.declare_parameter("num_obstacles", 3)
        self.declare_parameter("obstacle_types", ["person", "dog", "cat"])
        self.declare_parameter("spawn_area_radius", 3.0)
        self.declare_parameter("trajectory_radius", 1.5)
        self.declare_parameter("trajectory_speed", 0.6)  # rad/s
        self.declare_parameter("update_rate_hz", 20.0)

        self.world_name = str(self.get_parameter("world_name").value)
        self.num_obstacles = int(self.get_parameter("num_obstacles").value)
        self.obstacle_types = list(self.get_parameter("obstacle_types").value)
        self.spawn_area_radius = float(self.get_parameter("spawn_area_radius").value)
        self.trajectory_radius = float(self.get_parameter("trajectory_radius").value)
        self.trajectory_speed = float(self.get_parameter("trajectory_speed").value)
        self.update_rate_hz = float(self.get_parameter("update_rate_hz").value)

        # ---- Gazebo Sim services (after bridging)
        self.spawn_service = f"/world/{self.world_name}/create"
        self.delete_service = f"/world/{self.world_name}/remove"
        self.set_pose_service = f"/world/{self.world_name}/set_pose"

        self.spawn_client = self.create_client(SpawnEntity, self.spawn_service)
        self.delete_client = self.create_client(DeleteEntity, self.delete_service)
        self.set_pose_client = self.create_client(SetEntityPose, self.set_pose_service)

        self.get_logger().info(f"Waiting for services in world '{self.world_name}'...")
        for cli, srv_name in [
            (self.spawn_client, self.spawn_service),
            (self.delete_client, self.delete_service),
            (self.set_pose_client, self.set_pose_service),
        ]:
            if not cli.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(
                    f"Service {srv_name} not available. "
                    "Did you bridge it with ros_gz_bridge parameter_bridge?"
                )

        self.get_logger().info("Services available. Spawning obstacles...")

        # ---- State
        # name -> dict(type, center_x, center_y, angle, radius, speed, ready)
        self.obstacles = {}

        self._spawn_obstacles()

        self.dt = 1.0 / max(self.update_rate_hz, 1e-6)
        self.timer = self.create_timer(self.dt, self._update_trajectories)

        self.get_logger().info("DynamicObstacleSpawner running.")

    # ---------- Model creation helpers

    def _obstacle_spec(self, obstacle_type: str):
        colors = {
            "person": (1.0, 0.0, 0.0),
            "dog": (1.0, 0.5, 0.0),
            "cat": (1.0, 1.0, 0.0),
        }
        sizes = {
            "person": (0.4, 0.4, 1.7),
            "dog": (0.6, 0.3, 0.4),
            "cat": (0.3, 0.2, 0.3),
        }
        rgb = colors.get(obstacle_type, (1.0, 1.0, 1.0))
        size = sizes.get(obstacle_type, (0.5, 0.5, 0.5))
        return rgb, size

    def _make_sdf(self, model_name: str, obstacle_type: str) -> str:
        (r, g, b), (sx, sy, sz) = self._obstacle_spec(obstacle_type)

        return f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="{model_name}">
    <static>false</static>
    <link name="link">
      <pose>0 0 {sz/2.0} 0 0 0</pose>

      <visual name="visual">
        <geometry>
          <box><size>{sx} {sy} {sz}</size></box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
        </material>
      </visual>

      <collision name="collision">
        <geometry>
          <box><size>{sx} {sy} {sz}</size></box>
        </geometry>
      </collision>

      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.2</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.2</iyy><iyz>0</iyz>
          <izz>0.2</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>
"""

    # ---------- Spawn / move logic

    def _spawn_obstacles(self):
        for i in range(self.num_obstacles):
            obstacle_type = random.choice(self.obstacle_types)
            name = f"{obstacle_type}_{i}"

            cx = random.uniform(-self.spawn_area_radius, self.spawn_area_radius)
            cy = random.uniform(-self.spawn_area_radius, self.spawn_area_radius)
            angle0 = random.uniform(0.0, 2.0 * math.pi)

            pose = Pose()
            pose.position.x = float(cx)
            pose.position.y = float(cy)
            pose.position.z = 0.0
            pose.orientation.w = 1.0

            sdf = self._make_sdf(name, obstacle_type)

            req = SpawnEntity.Request()
            req.entity_factory = EntityFactory()
            req.entity_factory.name = name
            req.entity_factory.sdf = sdf
            req.entity_factory.pose = pose
            req.entity_factory.relative_to = "world"

            # Register obstacle immediately but mark not ready until spawn succeeds
            self.obstacles[name] = {
                "type": obstacle_type,
                "center_x": cx,
                "center_y": cy,
                "angle": angle0,
                "radius": self.trajectory_radius,
                "speed": self.trajectory_speed,
                "ready": False,  # <-- key change
            }

            future = self.spawn_client.call_async(req)
            future.add_done_callback(lambda f, n=name: self._log_spawn_result(f, n))

    def _log_spawn_result(self, future, name: str):
        try:
            resp = future.result()
            if resp.success:  # SpawnEntity returns bool success [web:258]
                if name in self.obstacles:
                    self.obstacles[name]["ready"] = True
                self.get_logger().info(f"Spawned: {name}")
            else:
                self.get_logger().error(f"Spawn failed: {name}")
        except Exception as e:
            self.get_logger().error(f"Spawn exception for {name}: {e}")

    def _update_trajectories(self):
        for name, obs in self.obstacles.items():
            # Don't move until spawn succeeded (prevents pose spam / errors)
            if not obs.get("ready", False):
                continue

            obs["angle"] += obs["speed"] * self.dt

            x = obs["center_x"] + obs["radius"] * math.cos(obs["angle"])
            y = obs["center_y"] + obs["radius"] * math.sin(obs["angle"])

            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0
            pose.orientation.w = 1.0

            req = SetEntityPose.Request()
            req.entity = Entity()
            req.entity.name = name
            req.entity.type = Entity.MODEL  # <-- critical for Gazebo to resolve entity [web:205]
            req.pose = pose

            self.set_pose_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Avoid "rcl_shutdown already called" during launch teardown
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
