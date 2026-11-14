#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("disable_client");

  auto client = node->create_client<std_srvs::srv::Trigger>(
    "/yaskawa_hc10/stop_traj_mode");

  // Wait for service
  while (!client->wait_for_service(std::chrono::seconds(2))) {
    RCLCPP_WARN(node->get_logger(), "Waiting for stop_traj_mode service...");
  }

  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();

  auto future = client->async_send_request(request);

  // Wait for response
  if (rclcpp::spin_until_future_complete(node, future) ==
      rclcpp::FutureReturnCode::SUCCESS)
  {
    auto response = future.get();

    if (response->success) {
      RCLCPP_INFO(node->get_logger(), "Trajectory mode DISABLED ✅");
    } else {
      RCLCPP_ERROR(node->get_logger(), "Service responded but failed ❌");
    }
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to call service ❌");
  }

  rclcpp::shutdown();
  return 0;
}
