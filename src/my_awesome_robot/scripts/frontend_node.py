#!/usr/bin/env python3
"""
Obstacle Detector Node
======================
Clusters raw LiDAR scan returns into obstacle groups and publishes:
  - /detected_obstacles  (PoseArray)   — obstacle centroids in lidar_link frame
  - /obstacle_markers    (MarkerArray) — RViz cylinder markers
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np


class FrontendNode(Node):
    def __init__(self):
        super().__init__('frontend_node')

        # Parameters
        self.declare_parameter('obstacle_distance_threshold', 2.0)
        self.declare_parameter('min_cluster_size', 3)
        self.declare_parameter('cluster_tolerance', 0.3)

        # Subscribers
        self._scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            10
        )

        # Publishers
        self._obstacle_pub = self.create_publisher(
            PoseArray,
            '/detected_obstacles',
            10
        )

        self._marker_pub = self.create_publisher(
            MarkerArray,
            '/obstacle_markers',
            10
        )

        self.get_logger().info('Frontend node initialized')

    def _scan_callback(self, msg: LaserScan) -> None:
        """Process laser scan data and detect obstacles."""
        obstacle_threshold = self.get_parameter('obstacle_distance_threshold').value
        min_cluster = self.get_parameter('min_cluster_size').value
        cluster_tol = self.get_parameter('cluster_tolerance').value

        # Convert laser scan to cartesian coordinates and cluster
        obstacles = []
        current_cluster = []

        for i, range_val in enumerate(msg.ranges):
            if range_val < msg.range_min or range_val > msg.range_max:
                continue

            if range_val < obstacle_threshold:
                angle = msg.angle_min + i * msg.angle_increment
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)

                if len(current_cluster) == 0:
                    current_cluster.append((x, y))
                else:
                    last_x, last_y = current_cluster[-1]
                    dist = np.linalg.norm([x - last_x, y - last_y])

                    if dist < cluster_tol:
                        current_cluster.append((x, y))
                    else:
                        if len(current_cluster) >= min_cluster:
                            obstacles.append(current_cluster)
                        current_cluster = [(x, y)]

        if len(current_cluster) >= min_cluster:
            obstacles.append(current_cluster)

        self._publish_obstacles(obstacles, msg.header)

    def _publish_obstacles(self, obstacles: list, header) -> None:
        """Publish detected obstacles as PoseArray and MarkerArray."""
        pose_array = PoseArray()
        pose_array.header = header
        pose_array.header.frame_id = 'lidar_link'

        marker_array = MarkerArray()

        for idx, cluster in enumerate(obstacles):
            x_mean = np.mean([p[0] for p in cluster])
            y_mean = np.mean([p[1] for p in cluster])

            pose = Pose()
            pose.position.x = x_mean
            pose.position.y = y_mean
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

            marker = Marker()
            marker.header = header
            marker.header.frame_id = 'lidar_link'
            marker.ns = 'obstacles'
            marker.id = idx
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker.lifetime.sec = 1
            marker_array.markers.append(marker)

        self._obstacle_pub.publish(pose_array)
        self._marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = FrontendNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
