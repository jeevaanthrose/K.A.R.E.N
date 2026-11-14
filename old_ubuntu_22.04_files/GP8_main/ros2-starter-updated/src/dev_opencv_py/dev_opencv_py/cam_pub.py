# Import the necessary libraries
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImagePublisher(Node):
    """
    ROS2 Node that publishes webcam images to 'video_frames'
    """

    def __init__(self):
        super().__init__('image_publisher')

        # Create publisher
        self.publisher_ = self.create_publisher(Image, 'video_frames', 10)

        # Publish at 10 Hz
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # -------- CAMERA SETUP (FIXED) --------
        camera_index = 0  # /dev/video5

        # Use V4L2 backend (important for Linux)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error('Failed to open camera!')
            raise RuntimeError('Cannot open camera')

        # Force MJPEG format (critical fix)
        self.cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*'MJPG')
        )

        # Set resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Warm-up camera (important for some devices)
        for _ in range(5):
            self.cap.read()

        # --------------------------------------

        self.br = CvBridge()
        self.frame_count = 0

        self.get_logger().info(
            'Camera publisher started on topic: video_frames'
        )

    def timer_callback(self):
        """
        Capture frame and publish
        """
        ret, frame = self.cap.read()

        if ret:
            msg = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)

            self.frame_count += 1
            if self.frame_count % 50 == 0:
                self.get_logger().info(
                    f'Publishing frame {self.frame_count}'
                )
        else:
            self.get_logger().warn('Failed to capture frame')

    def destroy_node(self):
        """
        Proper cleanup when node is destroyed
        """
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info('Camera released')

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    image_publisher = ImagePublisher()

    try:
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
