import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

import xacro


def generate_launch_description():

    # =========================================================
    # PACKAGE
    # =========================================================

    hc10_moveit_pkg = "hc10_moveit_config"

    # =========================================================
    # URDF
    # =========================================================

    hc10_urdf_config = xacro.process_file(
        os.path.join(
            get_package_share_directory(hc10_moveit_pkg),
            "config",
            "hc10.urdf.xacro",
        )
    )
    hc10_robot_description = {
        "robot_description": hc10_urdf_config.toxml()
    }

    # =========================================================
    # MOVEIT CONFIG
    # =========================================================

    moveit_config = (
        MoveItConfigsBuilder("hc10")
        .robot_description(file_path="config/hc10.urdf.xacro")
        .robot_description_semantic(file_path="config/hc10.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .to_moveit_configs()
    )

    # =========================================================
    # PLANNING SCENE MONITOR
    # =========================================================

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "publish_robot_description": True,
        "publish_robot_description_semantic": True,
    }

    # =========================================================
    # MOVE GROUP
    # trajectory_duration_monitoring disabled to prevent ABORTED
    # when FSU clamps execution speed below planned speed
    # =========================================================

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            planning_scene_monitor_parameters,
            {"trajectory_execution.trajectory_duration_monitoring": False},
        ],
    )

    # =========================================================
    # RVIZ
    # =========================================================

    rviz_config = os.path.join(
        get_package_share_directory(hc10_moveit_pkg),
        "config",
        "moveit.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
    )

    # =========================================================
    # JOINT STATE PUBLISHER
    # Sources from MotoROS2 topic /yaskawa_hc10/joint_states
    # =========================================================

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[
            {
                "robot_description": hc10_robot_description,
                "rate": 50,
                "source_list": ["/yaskawa_hc10/joint_states"],
            }
        ],
    )

    # =========================================================
    # ROBOT STATE PUBLISHER
    # Publishes TF from URDF
    # =========================================================

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            hc10_robot_description,
            {
                "publish_frequency": 50.0,
                "frame_prefix": "",
            },
        ],
    )

    # =========================================================
    # STATIC TF  (world → base_link)
    # Uncomment if your setup needs this transform
    # =========================================================

    # static_tf = Node(
    #     package="tf2_ros",
    #     executable="static_transform_publisher",
    #     name="static_transform_publisher",
    #     output="log",
    #     arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
    # )

    # =========================================================
    # LAUNCH DESCRIPTION
    # =========================================================

    return LaunchDescription([
        move_group_node,
        joint_state_publisher,
        robot_state_publisher,
        rviz_node,
    ])
