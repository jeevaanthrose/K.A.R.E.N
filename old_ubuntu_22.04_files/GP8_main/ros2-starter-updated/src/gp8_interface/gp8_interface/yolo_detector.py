#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Parameters
        self.declare_parameter('model', 'yolov8n.pt')  # nano model for speed
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('target_object', 'bottle')
        self.declare_parameter('end_effector_label', 'gripper')
        
        model_name = self.get_parameter('model').value
        self.confidence = self.get_parameter('confidence').value
        self.target_object = self.get_parameter('target_object').value
        self.end_effector_label = self.get_parameter('end_effector_label').value
        
        # Load YOLO model
        self.get_logger().info(f'Loading YOLO model: {model_name}')
        self.model = YOLO(model_name)
        
        # If you want to detect custom objects (like gripper), train a custom model
        # For now, we'll use standard COCO classes and assume gripper can be detected
        # or you can train a custom YOLOv8 model
        
        self.br = CvBridge()
        
        # Subscribe to camera
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.image_callback,
            10
        )
        
        # Publish detections
        self.detection_pub = self.create_publisher(String, 'yolo_detections', 10)
        
        # Publish annotated image
        self.annotated_pub = self.create_publisher(Image, 'yolo_annotated', 10)
        
        self.get_logger().info(f'YOLO detector started. Target: {self.target_object}')
        
    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence, verbose=False)
            
            # Parse detections
            detection_str = ""
            object_found = False
            effector_found = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.model.names[cls]
                    
                    # Convert to x, y, w, h format
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    # Check if it's our target object
                    if class_name.lower() == self.target_object.lower():
                        if detection_str:
                            detection_str += ";"
                        detection_str += f"{class_name}:{x},{y},{w},{h}"
                        object_found = True
                        
                        # Draw on frame
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} {conf:.2f}", 
                                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.6, (0, 255, 0), 2)
                    
                    # For end-effector detection, you'll need a custom-trained model
                    # For now, let's use a placeholder or detect specific shapes
                    # You can train YOLOv8 on images of your gripper
                    
            # Simple gripper detection based on color/shape (placeholder)
            # Replace this with actual YOLO detection if you train a custom model
            gripper_bbox = self.detect_gripper_simple(frame)
            if gripper_bbox is not None:
                x, y, w, h = gripper_bbox
                if detection_str:
                    detection_str += ";"
                detection_str += f"{self.end_effector_label}:{x},{y},{w},{h}"
                effector_found = True
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Gripper", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Publish detection string
            if detection_str:
                detection_msg = String()
                detection_msg.data = detection_str
                self.detection_pub.publish(detection_msg)
            
            # Publish annotated image
            annotated_msg = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
            self.annotated_pub.publish(annotated_msg)
            
            # Log status
            if object_found and effector_found:
                self.get_logger().debug("Both object and gripper detected")
            elif object_found:
                self.get_logger().debug("Object detected, gripper not found")
            elif effector_found:
                self.get_logger().debug("Gripper detected, object not found")
                
        except Exception as e:
            self.get_logger().error(f'Detection error: {str(e)}')
    
    def detect_gripper_simple(self, frame):
        """
        Simple gripper detection based on color/shape.
        Replace this with YOLO detection from a custom-trained model.
        
        For now, assuming gripper has a distinct color (e.g., metallic/gray)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect metallic/gray color (tune these values)
        lower = np.array([0, 0, 50])
        upper = np.array([180, 50, 200])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest) < 500:  # Minimum size
            return None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest)
        
        # Check aspect ratio (grippers are usually elongated)
        aspect = float(w) / h if h > 0 else 0
        if aspect < 0.3 or aspect > 3.0:  # Reasonable gripper shape
            return (x, y, w, h)
        
        return None

def main(args=None):
    rclpy.init(args=args)
    detector = YoloDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()