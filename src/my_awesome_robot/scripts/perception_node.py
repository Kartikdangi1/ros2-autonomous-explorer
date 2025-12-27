#!/usr/bin/env python3
"""
YOLOv8 Obstacle Detection Node for ROS2 Humble

Subscribes to camera feed, runs YOLOv8 inference, and publishes:
- Detected obstacles (custom message)
- Visualization markers for RViz
- Annotated image with bounding boxes

Author: Your Name
Date: 2025-12-27
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time


class PerceptionNode(Node):
    """ROS2 node for real-time obstacle detection using YOLOv8."""
    
    def __init__(self):
        super().__init__('perception_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('target_classes', [0, 2, 3, 5, 7])  # person, car, motorcycle, bus, truck
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('publish_annotated_image', True)
        self.declare_parameter('debug_mode', False)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.target_classes = self.get_parameter('target_classes').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.publish_annotated = self.get_parameter('publish_annotated_image').value
        self.debug_mode = self.get_parameter('debug_mode').value
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLOv8 model
        try:
            self.get_logger().info(f'Loading YOLOv8 model: {self.model_path}')
            self.model = YOLO(self.model_path)
            self.get_logger().info('YOLOv8 model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLOv8 model: {str(e)}')
            raise
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            10
        )
        
        # Publishers
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/obstacles',
            10
        )
        
        self.markers_pub = self.create_publisher(
            MarkerArray,
            '/obstacle_markers',
            10
        )
        
        if self.publish_annotated:
            self.annotated_image_pub = self.create_publisher(
                Image,
                '/camera/image_detections',
                10
            )
        
        # Performance metrics
        self.frame_count = 0
        self.total_inference_time = 0.0
        self.last_log_time = time.time()
        
        self.get_logger().info('Perception node initialized successfully')
        self.get_logger().info(f'Listening to: {self.camera_topic}')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')
        self.get_logger().info(f'Target classes: {self.target_classes}')
    
    def image_callback(self, msg: Image):
        """Process incoming camera images."""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run YOLOv8 inference
            start_time = time.time()
            results = self.model(
                cv_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.target_classes,
                verbose=False
            )
            inference_time = time.time() - start_time
            
            # Update performance metrics
            self.frame_count += 1
            self.total_inference_time += inference_time
            
            # Log performance every 5 seconds
            if time.time() - self.last_log_time > 5.0:
                avg_fps = self.frame_count / (time.time() - self.last_log_time)
                avg_inference = self.total_inference_time / self.frame_count * 1000
                self.get_logger().info(
                    f'Performance: {avg_fps:.1f} FPS | '
                    f'Avg inference: {avg_inference:.1f} ms'
                )
                self.frame_count = 0
                self.total_inference_time = 0.0
                self.last_log_time = time.time()
            
            # Process detections
            detection_array = self.process_detections(results[0], msg.header)
            
            # Publish detections
            self.detections_pub.publish(detection_array)
            
            # Publish visualization markers
            markers = self.create_markers(results[0], msg.header)
            self.markers_pub.publish(markers)
            
            # Publish annotated image
            if self.publish_annotated:
                annotated_image = results[0].plot()
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                annotated_msg.header = msg.header
                self.annotated_image_pub.publish(annotated_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def process_detections(self, result, header) -> Detection2DArray:
        """Convert YOLO results to ROS Detection2DArray."""
        detection_array = Detection2DArray()
        detection_array.header = header
        
        if result.boxes is None or len(result.boxes) == 0:
            return detection_array
        
        for box in result.boxes:
            detection = Detection2D()
            detection.header = header
            
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detection.bbox.center.position.x = float((x1 + x2) / 2)
            detection.bbox.center.position.y = float((y1 + y2) / 2)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            
            # Class and confidence
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(int(box.cls[0]))
            hypothesis.hypothesis.score = float(box.conf[0])
            detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)
        
        return detection_array
    
    def create_markers(self, result, header) -> MarkerArray:
        """Create RViz markers for detected obstacles."""
        marker_array = MarkerArray()
        
        # Delete old markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        if result.boxes is None or len(result.boxes) == 0:
            return marker_array
        
        for i, box in enumerate(result.boxes):
            marker = Marker()
            marker.header = header
            marker.header.frame_id = 'camera_optical_link'
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Position (simple depth estimation based on bbox size)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox_height = y2 - y1
            estimated_distance = 1.5 / (bbox_height / 480.0 + 0.1)  # Simple heuristic
            
            marker.pose.position.x = float(estimated_distance)
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Scale
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            
            # Color based on confidence
            confidence = float(box.conf[0])
            marker.color.r = 1.0
            marker.color.g = 1.0 - confidence
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 500000000  # 0.5 seconds
            
            marker_array.markers.append(marker)
        
        return marker_array


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {str(e)}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
