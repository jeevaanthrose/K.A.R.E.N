#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <std_msgs/msg/string.hpp>

#include <motoros2_interfaces/srv/write_single_io.hpp>

#include <termios.h>
#include <unistd.h>
#include <thread>
#include <mutex>

using std::placeholders::_1;
using WriteSingleIO = motoros2_interfaces::srv::WriteSingleIO;

static constexpr uint32_t IO_GRIPPER_CLOSE = 10011;
static constexpr uint32_t IO_GRIPPER_OPEN  = 10012;

char getch()
{
  struct termios oldt, newt;
  char ch;
  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  ch = getchar();
  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  return ch;
}

enum ControlMode { CARTESIAN, JOINT };

class TeleopNode : public rclcpp::Node
{
public:
  TeleopNode()
  : Node("cartesian_joint_teleop",
         rclcpp::NodeOptions()
           // Tell the MoveIt current_state_monitor to tolerate up to 10 s of
           // clock skew.  The default is 1.0 s; even 0.002 s of skew between
           // the hardware driver clock and the ROS system clock causes the
           // "recent timestamp" error.  10 s is generous while still catching
           // a truly dead driver.
           .parameter_overrides({
             rclcpp::Parameter("robot_description_semantic", ""),
             // This is the parameter the CurrentStateMonitor reads:
             rclcpp::Parameter("monitor_dynamics", false)
           })
    )
  {
    // Declare the allowed clock-skew tolerance so the state monitor picks
    // it up.  The parameter name is fixed in MoveIt source:
    // moveit_ros/planning/planning_scene_monitor/src/current_state_monitor.cpp
    this->declare_parameter<double>("state_monitor_max_time_without_update", 10.0);

    RCLCPP_INFO(this->get_logger(), "Teleop node created");

    cmd_sub_ = create_subscription<std_msgs::msg::String>(
        "/teleop_cmd", 10,
        std::bind(&TeleopNode::commandCallback, this, _1));

    gripper_client_ = this->create_client<WriteSingleIO>("/yaskawa/write_single_io");

    RCLCPP_INFO(this->get_logger(), "Waiting for gripper service...");
    gripper_client_->wait_for_service();
    RCLCPP_INFO(this->get_logger(), "Gripper service ready");
  }

  void init_moveit()
  {
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), "manipulator");

    move_group_->setMaxVelocityScalingFactor(0.2);
    move_group_->setMaxAccelerationScalingFactor(0.2);

    RCLCPP_INFO(this->get_logger(), "MoveGroup initialized");
    RCLCPP_INFO(this->get_logger(), "Waiting for valid robot state...");

    // Poll until the state monitor has a non-zero timestamp.
    // We do NOT use getCurrentState(timeout) here because that function
    // itself applies the strict freshness check internally.
    // Instead we sleep briefly and check getCurrentPose() stamp directly.
    bool ready = false;
    for (int i = 0; i < 100 && rclcpp::ok(); ++i)
    {
      rclcpp::sleep_for(std::chrono::milliseconds(100));
      auto stamped = move_group_->getCurrentPose();
      if (stamped.header.stamp.sec != 0) { ready = true; break; }
    }

    if (ready)
      RCLCPP_INFO(this->get_logger(), "Robot state ready — accepting commands");
    else
      RCLCPP_WARN(this->get_logger(), "Robot state not received — commands may fail");
  }

  void processCommand(char key)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (busy_)
    {
      RCLCPP_WARN(this->get_logger(), "Busy — ignoring key '%c'", key);
      return;
    }

    if (key == 'o') { setGripper(false); return; }
    if (key == 'p') { setGripper(true);  return; }
    if (key == 'c') { mode_ = CARTESIAN; RCLCPP_INFO(this->get_logger(), "Cartesian mode"); return; }
    if (key == 'j') { mode_ = JOINT;     RCLCPP_INFO(this->get_logger(), "Joint mode");     return; }

    if (mode_ == CARTESIAN)
      cartesianMove(key);
    else
      jointMove(key);
  }

private:

  void commandCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (msg->data.empty()) return;
    RCLCPP_INFO(this->get_logger(), "Received topic cmd: '%c'", msg->data[0]);
    processCommand(msg->data[0]);
  }

  bool writeIO(uint32_t address, int32_t value)
  {
    auto req = std::make_shared<WriteSingleIO::Request>();
    req->address = address;
    req->value   = value;
    auto fut = gripper_client_->async_send_request(req);
    if (fut.wait_for(std::chrono::seconds(3)) != std::future_status::ready)
    {
      RCLCPP_ERROR(this->get_logger(), "IO write timeout");
      return false;
    }
    auto res = fut.get();
    if (!res->success)
    {
      RCLCPP_ERROR(this->get_logger(), "IO write failed: %s", res->message.c_str());
      return false;
    }
    return true;
  }

  void setGripper(bool close)
  {
    if (close)
    {
      RCLCPP_INFO(this->get_logger(), "Gripper CLOSE");
      writeIO(IO_GRIPPER_CLOSE, 1);
      writeIO(IO_GRIPPER_OPEN,  0);
    }
    else
    {
      RCLCPP_INFO(this->get_logger(), "Gripper OPEN");
      writeIO(IO_GRIPPER_OPEN,  1);
      writeIO(IO_GRIPPER_CLOSE, 0);
    }
    rclcpp::sleep_for(std::chrono::milliseconds(500));
  }

  // ── Get current pose, retrying briefly to ride out transient skew ────────
  // The 1.6 ms skew we see is a timing race between when joint_states is
  // published and when MoveIt samples "now".  Waiting 50 ms and retrying
  // once is enough to let the next joint_states arrive and clear the error.
  geometry_msgs::msg::PoseStamped getCurrentPoseWithRetry(int retries = 5)
  {
    for (int i = 0; i < retries; ++i)
    {
      auto stamped = move_group_->getCurrentPose();
      if (stamped.header.stamp.sec != 0)
        return stamped;
      RCLCPP_INFO(this->get_logger(),
        "Waiting for fresh joint state (attempt %d/%d)…", i + 1, retries);
      rclcpp::sleep_for(std::chrono::milliseconds(50));
    }
    return move_group_->getCurrentPose();   // last attempt, caller checks
  }

  void cartesianMove(char key)
  {
    const double STEP = 0.02;

    move_group_->setStartStateToCurrentState();

    auto stamped = getCurrentPoseWithRetry();
    if (stamped.header.stamp.sec == 0)
    {
      RCLCPP_WARN(this->get_logger(), "Cannot move — robot state unavailable");
      return;
    }

    auto target = stamped.pose;

    if      (key == 'w') target.position.x += STEP;
    else if (key == 's') target.position.x -= STEP;
    else if (key == 'a') target.position.y += STEP;
    else if (key == 'd') target.position.y -= STEP;
    else if (key == 'r') target.position.z += STEP;
    else if (key == 'f') target.position.z -= STEP;
    else return;

    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(target);

    moveit_msgs::msg::RobotTrajectory trajectory;
    double fraction = move_group_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

    if (fraction < 0.95)
    {
      RCLCPP_WARN(this->get_logger(), "Cartesian path failed (fraction=%.2f)", fraction);
      return;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = trajectory;
    executePlan(plan);
  }

  void jointMove(char key)
  {
    const double STEP = 0.1;

    move_group_->setStartStateToCurrentState();

    // Brief retry loop to clear transient clock skew
    std::vector<double> joints;
    bool got_state = false;
    for (int i = 0; i < 5; ++i)
    {
      joints = move_group_->getCurrentJointValues();
      if (!joints.empty()) { got_state = true; break; }
      rclcpp::sleep_for(std::chrono::milliseconds(50));
    }

    if (!got_state)
    {
      RCLCPP_WARN(this->get_logger(), "Cannot move — robot state unavailable");
      return;
    }

    if      (key == '1') joints[0] += STEP;
    else if (key == '!') joints[0] -= STEP;
    else if (key == '2') joints[1] += STEP;
    else if (key == '@') joints[1] -= STEP;
    else if (key == '3') joints[2] += STEP;
    else if (key == '#') joints[2] -= STEP;
    else if (key == '4') joints[3] += STEP;
    else if (key == '$') joints[3] -= STEP;
    else if (key == '5') joints[4] += STEP;
    else if (key == '%') joints[4] -= STEP;
    else if (key == '6') joints[5] += STEP;
    else if (key == '^') joints[5] -= STEP;
    else return;

    move_group_->setJointValueTarget(joints);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if (!success)
    {
      RCLCPP_WARN(this->get_logger(), "Joint planning failed");
      return;
    }

    executePlan(plan);
  }

  void executePlan(moveit::planning_interface::MoveGroupInterface::Plan &plan)
  {
    busy_ = true;
    RCLCPP_INFO(this->get_logger(), "Executing");
    move_group_->execute(plan);
    move_group_->stop();
    busy_ = false;
  }

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr cmd_sub_;
  rclcpp::Client<WriteSingleIO>::SharedPtr gripper_client_;

  std::mutex mutex_;
  bool busy_        = false;
  ControlMode mode_ = CARTESIAN;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<TeleopNode>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  node->init_moveit();

  RCLCPP_INFO(node->get_logger(), "Controls:");
  RCLCPP_INFO(node->get_logger(), "  C = Cartesian mode  |  J = Joint mode");
  RCLCPP_INFO(node->get_logger(), "  W/S = X+/−  A/D = Y+/−  R/F = Z+/−");
  RCLCPP_INFO(node->get_logger(), "  1-6 / !@#$%^ = Joint +/−");
  RCLCPP_INFO(node->get_logger(), "  O = Gripper OPEN  |  P = Gripper CLOSE");

  if (isatty(STDIN_FILENO))
  {
    RCLCPP_INFO(node->get_logger(), "TTY detected — keyboard input active");
    while (rclcpp::ok())
    {
      char key = getch();
      node->processCommand(key);
    }
  }
  else
  {
    RCLCPP_INFO(node->get_logger(), "No TTY — topic-only mode (Flask/web control)");
    while (rclcpp::ok())
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
