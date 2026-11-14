#!/usr/bin/env python3

import rclpy
import sys
import termios
import tty
import math
import time

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from motoros2_interfaces.srv import StartTrajMode
from motoros2_interfaces.srv import QueueTrajPoint


STEP = math.radians(2.0)


class GP8JogWASD(Node):

    def __init__(self):
        super().__init__('gp8_jog_wasd')

        profile = QoSProfile(depth=1)
        profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        self.joint_pos = None

        self.create_subscription(
            JointState, 'joint_states', self.joint_cb, profile)

        # Start trajectory mode
        self.start_traj_mode()

        # Queue client
        self.queue_client = self.create_client(
            QueueTrajPoint, "/queue_traj_point")
        while not self.queue_client.wait_for_service(1.0):
            self.get_logger().info("Waiting for queue service...")

        self.req = QueueTrajPoint.Request()

        self.get_logger().info("READY FOR WASD JOG")

    def start_traj_mode(self):
        client = self.create_client(StartTrajMode, "yaskawa/start_traj_mode")
        while not client.wait_for_service(1.0):
            pass
        client.call_async(StartTrajMode.Request())

    def joint_cb(self, msg):
        self.joint_pos = list(msg.position)

    def get_key(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def send_queue_point(self, target):

        names = [
            'group_1/joint_1',
            'group_1/joint_2',
            'group_1/joint_3',
            'group_1/joint_4',
            'group_1/joint_5',
            'group_1/joint_6'
        ]

        pt = JointTrajectoryPoint()
        pt.positions = target
        pt.velocities = [0.5]*6
        pt.time_from_start = Duration(seconds=0.2).to_msg()

        self.req.joint_names = names
        self.req.point = pt

        self.queue_client.call_async(self.req)

    def run(self):

        print("""
W/S J2 | A/D J1 | Q/E J3 | Z/X J4 | R/F J5 | T/G J6 | ESC quit
""")

        while rclpy.ok():
            rclpy.spin_once(self)

            if self.joint_pos is None:
                continue

            key = self.get_key()
            target = list(self.joint_pos)

            if key == 'w': target[1] += STEP
            elif key == 's': target[1] -= STEP
            elif key == 'a': target[0] += STEP
            elif key == 'd': target[0] -= STEP
            elif key == 'q': target[2] += STEP
            elif key == 'e': target[2] -= STEP
            elif key == 'z': target[3] += STEP
            elif key == 'x': target[3] -= STEP
            elif key == 'r': target[4] += STEP
            elif key == 'f': target[4] -= STEP
            elif key == 't': target[5] += STEP
            elif key == 'g': target[5] -= STEP
            elif ord(key) == 27:
                break
            else:
                continue

            self.send_queue_point(target)
            time.sleep(0.15)


def main():
    rclpy.init()
    node = GP8JogWASD()
    node.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
