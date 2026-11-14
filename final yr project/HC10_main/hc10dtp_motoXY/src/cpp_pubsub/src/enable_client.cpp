#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "motoros2_interfaces/srv/start_traj_mode.hpp"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("enable_client");

  auto client = node->create_client<motoros2_interfaces::srv::StartTrajMode>(
    "/yaskawa_hc10/start_traj_mode");

  // Wait for service
  while (!client->wait_for_service(std::chrono::seconds(2))) {
    RCLCPP_WARN(node->get_logger(), "Waiting for service...");
  }

  auto request = std::make_shared<motoros2_interfaces::srv::StartTrajMode::Request>();

  auto future = client->async_send_request(request);

  // Wait for response
  if (rclcpp::spin_until_future_complete(node, future) ==
      rclcpp::FutureReturnCode::SUCCESS)
  {
    RCLCPP_INFO(node->get_logger(), "Trajectory mode ENABLED ✅");
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to call service ❌");
  }

  rclcpp::shutdown();
  return 0;
}
