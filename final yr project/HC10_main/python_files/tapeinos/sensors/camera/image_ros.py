"""
Minimal ROS Image <-> OpenCV helpers for bgr8 camera frames.

ROS Humble's cv_bridge binary is commonly built against NumPy 1.x.  If a
process accidentally imports NumPy 2.x first, cv_bridge can crash.  These
helpers avoid that dependency for the simple encodings used by Tapeinos.
"""

from __future__ import annotations

import cv2
import numpy as np
from sensor_msgs.msg import Image


def bgr8_to_imgmsg(frame: np.ndarray, frame_id: str = "") -> Image:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected a BGR image with shape HxWx3")
    if frame.dtype != np.uint8:
        raise ValueError("expected uint8 image data")

    contiguous = np.ascontiguousarray(frame)
    msg = Image()
    msg.header.frame_id = frame_id
    msg.height = int(contiguous.shape[0])
    msg.width = int(contiguous.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = int(contiguous.strides[0])
    msg.data = contiguous.tobytes()
    return msg


def imgmsg_to_bgr8(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding in ("bgr8", "rgb8"):
        channels = 3
    elif encoding in ("mono8", "8uc1"):
        channels = 1
    else:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")

    row_bytes = int(msg.step)
    expected = int(msg.height) * row_bytes
    data = np.frombuffer(msg.data, dtype=np.uint8, count=expected)
    rows = data.reshape((int(msg.height), row_bytes))
    pixels = rows[:, : int(msg.width) * channels]

    if channels == 1:
        mono = pixels.reshape((int(msg.height), int(msg.width)))
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    frame = pixels.reshape((int(msg.height), int(msg.width), channels)).copy()
    if encoding == "rgb8":
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame
