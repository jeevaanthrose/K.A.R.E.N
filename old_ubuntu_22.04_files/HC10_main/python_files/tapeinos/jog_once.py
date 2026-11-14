#!/usr/bin/env python3
"""
jog_once.py — MoveIt2 jog module for Tapeinos
===============================================
Two usage modes:

  1. IMPORTED by app.py (fast path):
       from jog_once import JogRunner
       runner = JogRunner()          # starts rclpy + spin thread once
       runner.execute('w')           # non-blocking, returns quickly
       runner.shutdown()             # call on app exit

  2. STANDALONE CLI (unchanged behaviour):
       python3 jog_once.py <key>

Cartesian : w s a d r f
Joint     : 1 2 3 4 5 6  (+step)    ! @ # $ % ^  (-step)
Mode      : c (cartesian)  j (joint)
Gripper   : o (suction on)  p (suction off)  — IO address 10017

Mode is persisted in /tmp/jog_state.json between calls.
"""

from __future__ import annotations

import sys
import json
import os
import time
import threading
import logging
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState
from industrial_msgs.msg import RobotStatus                      # type: ignore
from moveit_msgs.action import MoveGroup, ExecuteTrajectory      # type: ignore
from moveit_msgs.msg import (                                    # type: ignore
    Constraints, JointConstraint, MoveItErrorCodes, RobotState,
)
from moveit_msgs.srv import GetCartesianPath                     # type: ignore
from geometry_msgs.msg import Pose
import tf2_ros

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_FILE     = "/tmp/jog_state.json"
STEP_CART      = 0.02
STEP_JOINT     = 0.1
ROBOT_NAMESPACE = os.environ.get("TAPEINOS_ROBOT_NS", "/yaskawa_hc10").rstrip("/")
MOVE_GROUP_NAME = os.environ.get("TAPEINOS_MOVE_GROUP", "manipulator")


def _csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return values or default


# HC10 / MotoROS2 joint names observed on /yaskawa_hc10/joint_states:
# joint_1_s, joint_2_l, joint_3_u, joint_4_r, joint_5_b, joint_6_t.
JOINT_NAMES = _csv_env("TAPEINOS_JOINT_NAMES", [
    "joint_1_s", "joint_2_l", "joint_3_u",
    "joint_4_r", "joint_5_b", "joint_6_t",
])
_LEGACY_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6",
]
JOINT_ALIASES = {
    target: tuple(dict.fromkeys((target, legacy)))
    for target, legacy in zip(JOINT_NAMES, _LEGACY_JOINT_NAMES)
}
JOINT_STATE_TOPICS = _csv_env("TAPEINOS_JOINT_STATE_TOPICS", [
    f"{ROBOT_NAMESPACE}/joint_states",
    f"{ROBOT_NAMESPACE}/ctrl_groups/r1/joint_states",
    "/joint_states",
])
ROBOT_STATUS_TOPIC = os.environ.get(
    "TAPEINOS_ROBOT_STATUS_TOPIC",
    f"{ROBOT_NAMESPACE}/robot_status",
)
GRIPPER_IO_SERVICE = os.environ.get(
    "TAPEINOS_GRIPPER_IO_SERVICE",
    f"{ROBOT_NAMESPACE}/write_single_io",
)
MOVE_ACTION_TOPIC = os.environ.get("TAPEINOS_MOVE_ACTION", "/move_action")
EXECUTE_TRAJECTORY_TOPIC = os.environ.get(
    "TAPEINOS_EXECUTE_TRAJECTORY_ACTION",
    "/execute_trajectory",
)
COMPUTE_CARTESIAN_PATH_SERVICE = os.environ.get(
    "TAPEINOS_COMPUTE_CARTESIAN_PATH_SERVICE",
    "/compute_cartesian_path",
)
BASE_FRAME = os.environ.get("TAPEINOS_BASE_FRAME", "base_link")
EEF_CANDIDATES = _csv_env(
    "TAPEINOS_EEF_CANDIDATES",
    ["tool0", "flange", "link_6_t", "link_t", "link_6"],
)

# The Yaskawa driver publishes robot_status as Best Effort / Volatile.
# A Reliable subscriber is QoS-incompatible and receives nothing.
_BEST_EFFORT_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

JOINT_MAP = {
    '1': (0,  STEP_JOINT), '!': (0, -STEP_JOINT),
    '2': (1,  STEP_JOINT), '@': (1, -STEP_JOINT),
    '3': (2,  STEP_JOINT), '#': (2, -STEP_JOINT),
    '4': (3,  STEP_JOINT), '$': (3, -STEP_JOINT),
    '5': (4,  STEP_JOINT), '%': (4, -STEP_JOINT),
    '6': (5,  STEP_JOINT), '^': (5, -STEP_JOINT),
}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"mode": "cartesian"}


def save_state(s: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(s, f)


# ---------------------------------------------------------------------------
# JogNode
# ---------------------------------------------------------------------------

class JogNode(Node):
    def __init__(self) -> None:
        super().__init__("jog_once")
        self._js:   Optional[JointState] = None
        self._lock  = threading.Lock()

        # Robot status — tracks motion_possible / in_error etc.
        self._robot_status: Optional[RobotStatus] = None
        self._status_lock = threading.Lock()

        # Stop signals from ultrasonic sensors: topic -> bool
        self._stop_signals: dict[str, bool] = {}
        self._stop_lock = threading.Lock()

        # Back-reference to executor — set by JogRunner after creation
        self._executor = None

        self._js_score = -1
        for topic in JOINT_STATE_TOPICS:
            self.create_subscription(
                JointState,
                topic,
                lambda msg, t=topic: self._js_cb(msg, t),
                10,
            )
            self.get_logger().info(f"[hc10] listening for joint states on {topic}")

        # The Yaskawa driver publishes robot_status with Best Effort reliability.
        # Using the default Reliable QoS causes a QoS incompatibility and zero
        # messages are received.  Match the publisher with Best Effort + Volatile.
        self.create_subscription(
            RobotStatus, ROBOT_STATUS_TOPIC, self._status_cb,
            _BEST_EFFORT_QOS,
        )
        self.get_logger().info(f"[hc10] listening for robot status on {ROBOT_STATUS_TOPIC}")

        self._move_client = ActionClient(self, MoveGroup,         MOVE_ACTION_TOPIC)
        self._exec_client = ActionClient(self, ExecuteTrajectory, EXECUTE_TRAJECTORY_TOPIC)
        self._cart_client = self.create_client(
            GetCartesianPath,
            COMPUTE_CARTESIAN_PATH_SERVICE,
        )
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Gripper client created once at init — reused by every do_gripper() call.
        from motoros2_interfaces.srv import WriteSingleIO           # type: ignore
        self._gripper_client = self.create_client(
            WriteSingleIO, GRIPPER_IO_SERVICE
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _joint_match_score(self, msg: JointState) -> int:
        names = set(msg.name)
        return sum(1 for aliases in JOINT_ALIASES.values()
                   if any(alias in names for alias in aliases))

    def _js_cb(self, msg: JointState, topic: str = "") -> None:
        score = self._joint_match_score(msg)
        with self._lock:
            if self._js is not None and score < self._js_score:
                return
            self._js = msg
            self._js_score = score

    def _status_cb(self, msg: RobotStatus) -> None:
        with self._status_lock:
            self._robot_status = msg

    # ------------------------------------------------------------------
    # Robot readiness
    # ------------------------------------------------------------------

    def check_robot_ready(self, wait_timeout: float = 5.0) -> tuple[bool, str]:
        """Return (ready, reason).  Call before issuing any move.

        Waits up to *wait_timeout* seconds for the first robot_status message
        to arrive so transient startup races don't produce spurious failures.
        Once a message has been received the check is instantaneous.
        """
        deadline = time.time() + wait_timeout
        while True:
            with self._status_lock:
                s = self._robot_status
            if s is not None:
                break
            if time.time() >= deadline:
                return False, "no robot_status received (timed out waiting)"
            time.sleep(0.05)

        if s.e_stopped.val != 0:
            return False, "robot is E-stopped"
        if s.in_error.val != 0:
            return False, f"robot in error — codes: {s.error_codes}"
        if s.drives_powered.val == 0:
            return False, "drives not powered"
        if s.motion_possible.val == 0:
            return False, (
                "motion_possible=0 — put the pendant in REMOTE/AUTO mode "
                "and ensure no active alarms"
            )
        return True, ""

    # ------------------------------------------------------------------
    # Stop-signal API
    # ------------------------------------------------------------------

    def subscribe_stop_topic(self, topic: str) -> None:
        from std_msgs.msg import Bool
        with self._stop_lock:
            if topic in self._stop_signals:
                return
            self._stop_signals[topic] = False
        def _cb(msg, t=topic):
            with self._stop_lock:
                self._stop_signals[t] = msg.data
        self.create_subscription(Bool, topic, _cb, 10)
        self.get_logger().info(f"[stop-guard] subscribed to {topic}")

    def unsubscribe_stop_topic(self, topic: str) -> None:
        with self._stop_lock:
            self._stop_signals.pop(topic, None)

    def is_stopped(self) -> tuple[bool, str]:
        with self._stop_lock:
            for topic, active in self._stop_signals.items():
                if active:
                    return True, topic
        return False, ""

    # ------------------------------------------------------------------
    # Spinning helpers
    # ------------------------------------------------------------------

    def spin_until(self, future, timeout: float = 10.0,
                   abort_if_stopped: bool = False) -> bool:
        """
        Wait for *future* without re-entering the executor.
        Uses a threading.Event signalled by the future's done-callback so
        the executor thread keeps spinning freely while this thread waits.
        """
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())

        if future.done():
            done_event.set()

        deadline      = time.time() + timeout
        poll_interval = 0.05

        while True:
            if abort_if_stopped:
                stopped, topic = self.is_stopped()
                if stopped:
                    self.get_logger().warn(
                        f"[stop-guard] motion aborted — stop signal on {topic}")
                    return False
            if done_event.is_set():
                return future.done()
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            done_event.wait(timeout=min(poll_interval, remaining))

    def wait_for_joint_state(self, timeout: float = 10.0) -> Optional[JointState]:
        """Poll (don't spin_once) so this is safe when an executor owns the node."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._js is not None:
                    return self._js
            time.sleep(0.05)
        return None

    # ------------------------------------------------------------------
    # Joint-state helpers
    # ------------------------------------------------------------------

    def get_joint_values(self) -> Optional[list]:
        """Return the 6 arm joint positions in JOINT_NAMES order."""
        with self._lock:
            js = self._js
        if js is None:
            return None
        m = dict(zip(js.name, js.position))
        values = []
        missing = []
        for target_name in JOINT_NAMES:
            value = None
            for alias in JOINT_ALIASES.get(target_name, (target_name,)):
                if alias in m:
                    value = m[alias]
                    break
            if value is None:
                missing.append(target_name)
                value = 0.0
            values.append(value)
        if missing:
            self.get_logger().warn(
                "joint state missing HC10 joints: "
                f"{missing}; received: {list(js.name)}"
            )
            return None
        return values

    def make_robot_state(self, joint_values: list) -> RobotState:
        """
        Build a RobotState that MoveIt accepts as a valid start state.

        is_diff=True tells MoveIt to apply these joint values on top of its
        current internal state rather than requiring a complete state
        description.  Without this flag MoveIt rejects goals with
        INVALID_ROBOT_STATE (code 99999) because the message is missing the
        full kinematic chain (transforms, mimic joints, etc.) — exactly what
        the C++ setStartStateToCurrentState() fills in automatically.

        We seed the message from the live /joint_states so that any joints
        outside JOINT_NAMES (external axes, etc.) are included, then
        overwrite only the 6 arm joints with the caller's target values.
        """
        rs = RobotState()
        rs.is_diff = True  # critical — treat as delta over MoveIt's current state

        with self._lock:
            js = self._js

        if js is not None:
            # Start from the complete live joint state so no joints are absent.
            rs.joint_state.name     = list(js.name)
            rs.joint_state.position = list(js.position)
            if js.velocity:
                rs.joint_state.velocity = list(js.velocity)
            # Overwrite the 6 arm joints with the requested target positions.
            name_to_idx = {n: i for i, n in enumerate(rs.joint_state.name)}
            for name, val in zip(JOINT_NAMES, joint_values):
                if name in name_to_idx:
                    rs.joint_state.position[name_to_idx[name]] = float(val)
                else:
                    rs.joint_state.name.append(name)
                    rs.joint_state.position.append(float(val))
            if (rs.joint_state.velocity and
                    len(rs.joint_state.velocity) != len(rs.joint_state.name)):
                rs.joint_state.velocity = []
        else:
            # Fallback: 6-joint stub — is_diff covers the rest.
            rs.joint_state.name     = list(JOINT_NAMES)
            rs.joint_state.position = [float(v) for v in joint_values]

        return rs

    # ------------------------------------------------------------------
    # EEF pose
    # ------------------------------------------------------------------

    def get_eef_pose(self):
        """Look up the end-effector pose from TF, trying each candidate link."""
        time.sleep(0.3)
        for link in EEF_CANDIDATES:
            try:
                t = self._tf_buffer.lookup_transform(
                    BASE_FRAME, link,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5),
                )
                self.get_logger().info(f"EEF link: {link}")
                pose = Pose()
                pose.position.x    = t.transform.translation.x
                pose.position.y    = t.transform.translation.y
                pose.position.z    = t.transform.translation.z
                pose.orientation.x = t.transform.rotation.x
                pose.orientation.y = t.transform.rotation.y
                pose.orientation.z = t.transform.rotation.z
                pose.orientation.w = t.transform.rotation.w
                return pose, link
            except tf2_ros.TransformException:
                continue
        return None, None

    # ------------------------------------------------------------------
    # Goal cancellation
    # ------------------------------------------------------------------

    def _cancel_and_drain(self, gh, label: str = "") -> None:
        """Cancel a goal handle and wait up to 2 s for acknowledgement."""
        try:
            cancel_fut = gh.cancel_goal_async()
            done_event = threading.Event()
            cancel_fut.add_done_callback(lambda _: done_event.set())
            if cancel_fut.done():
                done_event.set()
            done_event.wait(timeout=2.0)
            time.sleep(0.15)
            if label:
                self.get_logger().info(f"[cancel] {label} cancelled and drained")
        except Exception as exc:
            self.get_logger().warn(f"[cancel] drain error: {exc}")

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_joints(self, joint_values: list, timeout: float = 30.0) -> bool:
        """Plan and execute a joint-space move to *joint_values*.

        Does NOT call check_robot_ready() — the caller (JogRunner.execute)
        already did that once, centrally, to avoid double 5-second waits.
        Does NOT pass an explicit start_state — letting MoveIt resolve it
        from its own scene monitor avoids the INVALID_ROBOT_STATE (99999)
        errors that an incomplete RobotState message caused previously.
        """
        if not self._move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup action server not available")
            return False

        goal = MoveGroup.Goal()
        req  = goal.request
        req.group_name                      = MOVE_GROUP_NAME
        req.num_planning_attempts           = 5
        req.allowed_planning_time           = 10.0
        req.max_velocity_scaling_factor     = 0.2
        req.max_acceleration_scaling_factor = 0.2

        c = Constraints()
        for name, val in zip(JOINT_NAMES, joint_values):
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(val)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)

        fut = self._move_client.send_goal_async(goal)
        if not self.spin_until(fut, timeout=5.0):
            self.get_logger().error("Goal send timed out")
            return False
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error("Goal rejected")
            return False

        res_fut = gh.get_result_async()
        if not self.spin_until(res_fut, timeout=timeout, abort_if_stopped=True):
            self.get_logger().warn("Move interrupted — cancelling goal")
            self._cancel_and_drain(gh, "move_joints")
            return False

        ok = res_fut.result().result.error_code.val == MoveItErrorCodes.SUCCESS
        if not ok:
            self.get_logger().warn(
                f"Move failed, code={res_fut.result().result.error_code.val}")
        return ok

    def move_cartesian(self, joint_values: list, key: str, timeout: float = 30.0) -> bool:
        """Compute a Cartesian step in direction *key* and execute it.

        Passes is_diff=True in the start_state (via make_robot_state) so MoveIt
        resolves it against its current scene — equivalent to the C++ node's
        setStartStateToCurrentState().  This fixes the 10-second planning timeout
        that occurred when an incomplete start state was sent previously.
        """
        pose, eef_link = self.get_eef_pose()
        if pose is None:
            self.get_logger().error(
                f"TF lookup failed for all candidates: {EEF_CANDIDATES}")
            return False

        if   key == 'w': pose.position.x += STEP_CART
        elif key == 's': pose.position.x -= STEP_CART
        elif key == 'a': pose.position.y += STEP_CART
        elif key == 'd': pose.position.y -= STEP_CART
        elif key == 'r': pose.position.z += STEP_CART
        elif key == 'f': pose.position.z -= STEP_CART

        if not self._cart_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/compute_cartesian_path not available")
            return False

        req = GetCartesianPath.Request()
        req.header.frame_id  = BASE_FRAME
        req.start_state      = self.make_robot_state(joint_values)
        req.group_name       = MOVE_GROUP_NAME
        req.link_name        = eef_link
        req.waypoints        = [pose]
        req.max_step         = 0.01
        req.jump_threshold   = 0.0
        req.avoid_collisions = True

        fut = self._cart_client.call_async(req)
        if not self.spin_until(fut, timeout=10.0):
            self.get_logger().error("/compute_cartesian_path timed out")
            return False

        res = fut.result()
        if res.fraction < 0.95:
            self.get_logger().warn(
                f"Cartesian path fraction too low: {res.fraction:.2f}")
            return False

        self.get_logger().info(
            f"Path computed (fraction={res.fraction:.2f}), executing…")

        if not self._exec_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("/execute_trajectory not available")
            return False

        exec_goal            = ExecuteTrajectory.Goal()
        exec_goal.trajectory = res.solution

        fut2 = self._exec_client.send_goal_async(exec_goal)
        if not self.spin_until(fut2, timeout=5.0):
            self.get_logger().error("execute_trajectory send timed out")
            return False
        gh2 = fut2.result()
        if not gh2.accepted:
            self.get_logger().error("execute_trajectory rejected")
            return False

        res_fut2 = gh2.get_result_async()
        if not self.spin_until(res_fut2, timeout=timeout, abort_if_stopped=True):
            self.get_logger().warn("execute_trajectory interrupted — cancelling goal")
            self._cancel_and_drain(gh2, "execute_trajectory")
            return False

        ok = res_fut2.result().result.error_code.val == MoveItErrorCodes.SUCCESS
        if not ok:
            self.get_logger().warn(
                f"execute_trajectory failed, "
                f"code={res_fut2.result().result.error_code.val}")
        return ok

    def do_gripper(self, suction_on: bool) -> bool:
        """Control the suction gripper via IO address 10017.
        suction_on=True  -> suction ON  (key 'o')
        suction_on=False -> suction OFF (key 'p')
        """
        if not self._gripper_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("gripper service not available")
            return False

        from motoros2_interfaces.srv import WriteSingleIO           # type: ignore

        req         = WriteSingleIO.Request()
        req.address = 10017
        req.value   = 1 if suction_on else 0

        label = "SUCTION ON" if suction_on else "SUCTION OFF"
        self.get_logger().info(f"gripper {label} (IO 10017 = {req.value})")

        fut = self._gripper_client.call_async(req)
        self.spin_until(fut, timeout=3.0)

        if not fut.done() or not fut.result().success:
            self.get_logger().error(f"gripper {label} -- IO write failed")
            return False
        return True


# ---------------------------------------------------------------------------
# JogRunner  -- persistent, importable, used by app.py
# ---------------------------------------------------------------------------

class JogRunner:
    """
    Owns a single rclpy context + JogNode + background spin thread.
    Call execute(key) from any thread; it blocks until the motion completes
    (or errors) but does NOT restart rclpy each time.

    Lifecycle::
        runner = JogRunner()   # call once at app startup
        runner.execute('w')    # call per jog command
        runner.shutdown()      # call at app exit
    """

    def __init__(self) -> None:
        self._lock     = threading.Lock()   # serialise concurrent jog calls
        self._ready    = threading.Event()
        self._node:    Optional[JogNode] = None
        self._executor = None
        self._thread:  Optional[threading.Thread] = None
        self._active   = False
        self._pending_stop_topics: set = set()
        self._pending_lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._spin_loop,
            daemon=True,
            name="jog-runner-spin",
        )
        self._thread.start()
        if not self._ready.wait(timeout=15.0):
            log.warning("[JogRunner] timed out waiting for joint state -- "
                        "commands will still work once ROS2 is up")

    # ------------------------------------------------------------------
    # Background spin loop
    # ------------------------------------------------------------------

    def _spin_loop(self) -> None:
        owns_rclpy = False
        try:
            if not rclpy.ok():
                rclpy.init()
                owns_rclpy = True

            self._node     = JogNode()
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._node._executor = self._executor
            self._active   = True
            self._flush_pending_topics()

            # Signal ready once a joint state has been received (or after 10 s).
            deadline = time.time() + 10.0
            while time.time() < deadline and self._active:
                self._executor.spin_once(timeout_sec=0.05)
                if self._node._js is not None:
                    break
            self._ready.set()

            self._executor.spin()

        except Exception as exc:
            log.error(f"[JogRunner] spin loop error: {exc}")
            self._ready.set()
        finally:
            try:
                if self._executor and self._node:
                    drain_end = time.time() + 0.5
                    while time.time() < drain_end:
                        self._executor.spin_once(timeout_sec=0.05)
                if self._node:
                    self._node.destroy_node()
                if owns_rclpy:
                    rclpy.shutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, key: str) -> tuple[bool, str]:
        """
        Execute one jog command synchronously (blocks caller's thread).
        Returns (success: bool, message: str).
        """
        if not self._active or self._node is None:
            return False, "JogRunner not initialised"

        key = key[0]

        # Mode switches -- pure state, no motion or readiness check needed.
        if key == 'c':
            state = load_state(); state["mode"] = "cartesian"; save_state(state)
            return True, "mode -> cartesian"
        if key == 'j':
            state = load_state(); state["mode"] = "joint";     save_state(state)
            return True, "mode -> joint"

        with self._lock:
            stopped, blocker = self._node.is_stopped()
            if stopped:
                return False, f"BLOCKED by stop signal on {blocker}"

            # Gripper is pure IO -- does NOT require motion_possible.
            if key in ('o', 'p'):
                ok = self._node.do_gripper(suction_on=(key == 'o'))
                return ok, "suction ON" if key == 'o' else "suction OFF"

            # All motion commands: check robot readiness once here, centrally.
            # move_joints / move_cartesian do NOT repeat this check so there
            # is no double 5-second wait on a stale status.
            ready, reason = self._node.check_robot_ready()
            if not ready:
                return False, f"Robot not ready: {reason}"

            joints = self._node.get_joint_values()
            if joints is None:
                return False, "no joint state received"

            if key in JOINT_MAP:
                idx, delta = JOINT_MAP[key]
                joints[idx] += delta
                msg = f"joint J{idx+1} {'+' if delta > 0 else ''}{delta:.2f} rad"
                log.info(f"[JogRunner] {msg}")
                ok = self._node.move_joints(joints)
                return ok, msg

            if key in "wsadrf":
                log.info(f"[JogRunner] cartesian '{key}'")
                ok = self._node.move_cartesian(joints, key)
                return ok, f"cartesian '{key}'"

            return False, f"unknown key '{key}'"

    def register_stop_topic(self, topic: str) -> None:
        if self._node is not None:
            self._node.subscribe_stop_topic(topic)
            log.info(f"[JogRunner] stop-guard registered: {topic}")
        else:
            with self._pending_lock:
                self._pending_stop_topics.add(topic)
            log.info(f"[JogRunner] stop-guard queued (node not ready): {topic}")

    def _flush_pending_topics(self) -> None:
        with self._pending_lock:
            topics = list(self._pending_stop_topics)
            self._pending_stop_topics.clear()
        for topic in topics:
            self._node.subscribe_stop_topic(topic)
            log.info(f"[JogRunner] stop-guard flushed: {topic}")

    def unregister_stop_topic(self, topic: str) -> None:
        with self._pending_lock:
            self._pending_stop_topics.discard(topic)
        if self._node is not None:
            self._node.unsubscribe_stop_topic(topic)

    def shutdown(self) -> None:
        """Stop the spin loop and clean up rclpy."""
        self._active = False
        if self._executor:
            self._executor.shutdown()
        if self._thread:
            self._thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_runner: Optional[JogRunner] = None
_runner_lock = threading.Lock()


def get_runner() -> JogRunner:
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = JogRunner()
    return _runner


def shutdown_runner() -> None:
    global _runner
    if _runner is not None:
        _runner.shutdown()
        _runner = None


# ---------------------------------------------------------------------------
# Standalone CLI  (python3 jog_once.py <key>)
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: jog_once.py <key>", file=sys.stderr)
        sys.exit(1)

    key   = sys.argv[1][0]
    state = load_state()

    if key == 'c':
        state["mode"] = "cartesian"; save_state(state)
        print("mode -> cartesian"); return
    if key == 'j':
        state["mode"] = "joint"; save_state(state)
        print("mode -> joint"); return

    rclpy.init()
    node = JogNode()

    # Standalone mode: no executor owns the node, so drive it with spin_once.
    print("waiting for joint state...")
    deadline = time.time() + 10.0
    got_js = False
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node._js is not None:
            got_js = True
            break
    if not got_js:
        print("ERROR: no joint state received", file=sys.stderr)
        node.destroy_node(); rclpy.shutdown(); sys.exit(1)

    # Gripper is pure IO -- handle before the motion-ready check.
    if key in ('o', 'p'):
        deadline2 = time.time() + 3.0
        while time.time() < deadline2:
            rclpy.spin_once(node, timeout_sec=0.05)
        ok = node.do_gripper(suction_on=(key == 'o'))
        node.destroy_node(); rclpy.shutdown(); sys.exit(0 if ok else 1)

    # Motion commands need robot_status -- spin until it arrives.
    print("waiting for robot status...")
    deadline3 = time.time() + 5.0
    while time.time() < deadline3:
        rclpy.spin_once(node, timeout_sec=0.05)
        with node._status_lock:
            if node._robot_status is not None:
                break

    # wait_timeout=0.0: we already spun above so _robot_status is populated
    # if it's coming.  The function returns timeout-error only if still None.
    ready, reason = node.check_robot_ready(wait_timeout=0.0)
    if not ready:
        print(f"ERROR: robot not ready -- {reason}", file=sys.stderr)
        node.destroy_node(); rclpy.shutdown(); sys.exit(1)

    joints = node.get_joint_values()

    if key in JOINT_MAP:
        idx, delta = JOINT_MAP[key]
        joints[idx] += delta
        print(f"joint J{idx+1} {'+' if delta > 0 else ''}{delta:.2f} rad")
        ok = node.move_joints(joints)
    elif key in "wsadrf":
        print(f"cartesian '{key}'")
        ok = node.move_cartesian(joints, key)
    else:
        print(f"unknown key '{key}'", file=sys.stderr)
        node.destroy_node(); rclpy.shutdown(); sys.exit(1)

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
