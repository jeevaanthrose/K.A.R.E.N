cmake_minimum_required(VERSION 3.8)
project(hello_moveit)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------- Dependencies ----------------
find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)

find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(control_msgs REQUIRED)

find_package(moveit_ros_planning_interface REQUIRED)
find_package(motoros2_interfaces REQUIRED)

# Vision dependencies
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ---------------- Executables ----------------

# Simulation example
add_executable(sim_hello_moveit src/sim_hello_moveit.cpp)
ament_target_dependencies(sim_hello_moveit
  motoros2_interfaces
  moveit_ros_planning_interface
  rclcpp
  std_srvs
  rclcpp_action
  rclcpp_components
  control_msgs
  std_msgs
  sensor_msgs
)

# Basic MoveIt example
add_executable(hello_moveit src/hello_moveit.cpp)
ament_target_dependencies(hello_moveit
  motoros2_interfaces
  moveit_ros_planning_interface
  rclcpp
  std_srvs
  rclcpp_action
  rclcpp_components
  control_msgs
  std_msgs
  sensor_msgs
)

# Joint motion example
add_executable(joint_hello_moveit src/joint_hello_moveit.cpp)
ament_target_dependencies(joint_hello_moveit
  motoros2_interfaces
  moveit_ros_planning_interface
  rclcpp
  std_srvs
  rclcpp_action
  rclcpp_components
  control_msgs
  std_msgs
  sensor_msgs
)

# Obstacle example
add_executable(hello_obstacle src/hello_obstacle.cpp)
ament_target_dependencies(hello_obstacle
  motoros2_interfaces
  moveit_ros_planning_interface
  rclcpp
  std_srvs
  rclcpp_action
  rclcpp_components
  control_msgs
  std_msgs
  sensor_msgs
)

# WASD Teleop
add_executable(wasd src/wasd.cpp)
ament_target_dependencies(wasd
  motoros2_interfaces
  moveit_ros_planning_interface
  rclcpp
  std_srvs
  rclcpp_action
  rclcpp_components
  control_msgs
  std_msgs
  sensor_msgs
  geometry_msgs
)

# ---------------- Green Tracker (Vision + MoveIt) ----------------

add_executable(green_tracker_moveit src/green_tracker_moveit.cpp)
ament_target_dependencies(green_tracker_moveit
  motoros2_interfaces
  moveit_ros_planning_interface
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
)
target_link_libraries(green_tracker_moveit
  ${OpenCV_LIBS}
)

# ---------------- Voice Assistant (Vision + MoveIt + IO) ----------------

add_executable(voice_assistant src/voice_assistant.cpp)
ament_target_dependencies(voice_assistant
  motoros2_interfaces
  moveit_ros_planning_interface
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
)
target_link_libraries(voice_assistant
  ${OpenCV_LIBS}
)

# ---------------- Install ----------------

install(TARGETS
  sim_hello_moveit
  hello_moveit
  joint_hello_moveit
  hello_obstacle
  wasd
  green_tracker_moveit
  voice_assistant
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------- Package ----------------
ament_package()
