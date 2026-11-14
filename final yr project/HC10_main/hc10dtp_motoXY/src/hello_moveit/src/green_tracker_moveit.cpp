#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <motoros2_interfaces/srv/start_traj_mode.hpp>
#include <motoros2_interfaces/srv/write_single_io.hpp>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <thread>
#include <mutex>
#include <string>
#include <cmath>

using moveit::planning_interface::MoveGroupInterface;
using WriteSingleIO = motoros2_interfaces::srv::WriteSingleIO;

// =========================================================
// USER SETTINGS
// =========================================================

// Minimum contour area (px²) to be considered a valid cylinder
static constexpr double MIN_CONTOUR_AREA = 100.0;

// ── Hough Circle settings ─────────────────────────────────
static constexpr double HOUGH_DP           = 1.2;
static constexpr double HOUGH_MIN_DIST     = 50.0;
static constexpr double HOUGH_PARAM1       = 100.0;
static constexpr double HOUGH_PARAM2       = 0.75;
static constexpr double HOUGH_RADIUS_FRACTION  = 0.45;
static constexpr double HOUGH_RADIUS_TOLERANCE = 0.40;
static constexpr double HOUGH_CENTRE_GATE  = 0.55;

// ── Suction gripper IO ────────────────────────────────────
// IO #10017: write 1 = suction ON (pick), write 0 = suction OFF (release)
static constexpr uint32_t IO_SUCTION = 10017;

// How far to lift after grasping (metres)
static constexpr double LIFT_HEIGHT = 0.08;

// How far to retreat upward after placing (metres)
static constexpr double RETREAT_HEIGHT = 0.06;

// Place offset from pick XY (metres)
static constexpr double PLACE_OFFSET_X =  0.08;
static constexpr double PLACE_OFFSET_Y =  0.08;

// =========================================================
// Load homography
// =========================================================
bool loadHomography(const std::string& path, cv::Mat& H)
{
  std::ifstream file(path);
  if (!file.is_open()) return false;
  H = cv::Mat(3, 3, CV_64F);
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      file >> H.at<double>(r, c);
  return file.good() || file.eof();
}

// =========================================================
// Camera Subscriber
// =========================================================
class FrameSubscriber : public rclcpp::Node
{
public:
  FrameSubscriber() : Node("frame_subscriber")
  {
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/video_frames", 10,
      [this](sensor_msgs::msg::Image::SharedPtr msg)
      {
        try {
          auto cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
          cv::Mat resized;
          cv::resize(cv_ptr->image, resized, cv::Size(640, 480));
          std::lock_guard<std::mutex> lock(mutex_);
          frame_ = resized;
        } catch (...) {}
      });
  }

  bool getFrame(cv::Mat& out)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frame_.empty()) return false;
    out = frame_.clone();
    return true;
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  cv::Mat frame_;
  std::mutex mutex_;
};

// =========================================================
// TopFaceResult
// =========================================================
struct TopFaceResult
{
  int    px = 0, py = 0;
  double radius   = 0.0;
  double axis_major = 0.0;
  double axis_minor = 0.0;
  double angle_deg  = 0.0;
  enum class Method { HOUGH, HULL, BBOX } method = Method::BBOX;
};

// =========================================================
// Cylinder top-face detection
// =========================================================
bool detectCylinderTop(const cv::Mat& frame, TopFaceResult& result)
{
  cv::Mat hsv, mask;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  cv::Mat mask1, mask2;
  cv::inRange(hsv, cv::Scalar(0,   100, 100), cv::Scalar(10,  255, 255), mask1);
  cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), mask2);
  cv::bitwise_or(mask1, mask2, mask);
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat::ones(5, 5, CV_8U));

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return false;

  auto best_it = std::max_element(contours.begin(), contours.end(),
      [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return cv::contourArea(a) < cv::contourArea(b);
      });
  if (cv::contourArea(*best_it) < MIN_CONTOUR_AREA) return false;
  const auto& contour = *best_it;

  cv::Rect bbox = cv::boundingRect(contour);

  const int pad = 10;
  cv::Rect roi_rect(
      std::max(0, bbox.x - pad),
      std::max(0, bbox.y - pad),
      std::min(frame.cols - std::max(0, bbox.x - pad), bbox.width  + 2*pad),
      std::min(frame.rows - std::max(0, bbox.y - pad), bbox.height + 2*pad));

  cv::Mat roi_mask  = mask(roi_rect);
  cv::Mat roi_gray;
  cv::cvtColor(frame(roi_rect), roi_gray, cv::COLOR_BGR2GRAY);
  roi_gray.setTo(0, ~roi_mask);
  cv::GaussianBlur(roi_gray, roi_gray, {9,9}, 2.0);

  double expected_r = bbox.width * HOUGH_RADIUS_FRACTION;
  int    r_min      = std::max(5, static_cast<int>(expected_r * (1.0 - HOUGH_RADIUS_TOLERANCE)));
  int    r_max      = static_cast<int>(expected_r * (1.0 + HOUGH_RADIUS_TOLERANCE));

  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(roi_gray, circles,
                   cv::HOUGH_GRADIENT_ALT,
                   HOUGH_DP, HOUGH_MIN_DIST,
                   HOUGH_PARAM1, HOUGH_PARAM2,
                   r_min, r_max);

  double bbox_cx_roi = (bbox.x - roi_rect.x) + bbox.width  / 2.0;
  double bbox_cy_roi = (bbox.y - roi_rect.y) + bbox.height / 2.0;
  double gate_x      = bbox.width  * HOUGH_CENTRE_GATE;
  double gate_y      = bbox.height * HOUGH_CENTRE_GATE;

  for (const auto& c : circles)
  {
    double cx_roi = c[0], cy_roi = c[1], r = c[2];
    if (std::abs(cx_roi - bbox_cx_roi) < gate_x &&
        std::abs(cy_roi - bbox_cy_roi) < gate_y)
    {
      result.px         = static_cast<int>(cx_roi + roi_rect.x);
      result.py         = static_cast<int>(cy_roi + roi_rect.y);
      result.radius     = r;
      result.axis_major = r;
      result.axis_minor = r;
      result.angle_deg  = 0.0;
      result.method     = TopFaceResult::Method::HOUGH;
      return true;
    }
  }

  std::vector<cv::Point> hull;
  cv::convexHull(contour, hull);

  if (hull.size() >= 5)
  {
    cv::RotatedRect ell = cv::fitEllipse(hull);
    result.px         = static_cast<int>(ell.center.x);
    result.py         = static_cast<int>(ell.center.y);
    result.radius     = (ell.size.width + ell.size.height) / 4.0;
    result.axis_major = ell.size.width  / 2.0;
    result.axis_minor = ell.size.height / 2.0;
    result.angle_deg  = ell.angle;
    result.method     = TopFaceResult::Method::HULL;
    return true;
  }

  result.px         = bbox.x + bbox.width  / 2;
  result.py         = bbox.y + bbox.height / 2;
  result.radius     = bbox.width / 2.0;
  result.axis_major = bbox.width  / 2.0;
  result.axis_minor = bbox.height / 2.0;
  result.angle_deg  = 0.0;
  result.method     = TopFaceResult::Method::BBOX;
  return true;
}

// =========================================================
// Draw top-face overlay
// =========================================================
void drawTopFaceOverlay(cv::Mat& frame, const TopFaceResult& r,
                        double rx, double ry)
{
  cv::Scalar colour =
      (r.method == TopFaceResult::Method::HOUGH) ? cv::Scalar(0, 220,  0)  :
      (r.method == TopFaceResult::Method::HULL)  ? cv::Scalar(0, 165, 255) :
                                                    cv::Scalar(0,   0, 255);

  if (r.method == TopFaceResult::Method::HOUGH)
  {
    cv::circle(frame, {r.px, r.py}, static_cast<int>(r.radius), colour, 2);
    int cr = static_cast<int>(r.radius);
    cv::line(frame, {r.px - cr, r.py}, {r.px + cr, r.py}, colour, 1);
    cv::line(frame, {r.px, r.py - cr}, {r.px, r.py + cr}, colour, 1);
  }
  else
  {
    if (r.axis_major > 1.0)
      cv::ellipse(frame, cv::Point(r.px, r.py),
                  cv::Size(static_cast<int>(r.axis_major),
                           static_cast<int>(r.axis_minor)),
                  r.angle_deg, 0, 360, colour, 2);

    if (r.method == TopFaceResult::Method::HULL && r.axis_major > 1.0)
    {
      double cos_a = std::cos(r.angle_deg * CV_PI / 180.0);
      double sin_a = std::sin(r.angle_deg * CV_PI / 180.0);
      cv::line(frame,
               {r.px - static_cast<int>(r.axis_major * cos_a),
                r.py - static_cast<int>(r.axis_major * sin_a)},
               {r.px + static_cast<int>(r.axis_major * cos_a),
                r.py + static_cast<int>(r.axis_major * sin_a)},
               {0, 255, 255}, 1);
    }
  }

  cv::circle(frame, {r.px, r.py}, 5, colour, -1);
  cv::circle(frame, {r.px, r.py}, 16, colour, 1);

  const char* tag =
      (r.method == TopFaceResult::Method::HOUGH) ? "[H]" :
      (r.method == TopFaceResult::Method::HULL)  ? "[E]" : "[B]";

  std::string label =
      std::string(tag) +
      " Top:(" + std::to_string(r.px) + "," + std::to_string(r.py) + ")"
      "  R:(" + std::to_string(rx).substr(0,6) +
      "," + std::to_string(ry).substr(0,6) + ")"
      "  r=" + std::to_string(static_cast<int>(r.radius)) + "px";

  cv::putText(frame, label, {r.px + 20, r.py - 10},
              cv::FONT_HERSHEY_SIMPLEX, 0.42, colour, 1);
}

// =========================================================
// Suction gripper control via /yaskawa_hc10/write_single_io
//
//   suctionOn  = true  → IO #10017 = 1  (suction ON,  pick)
//   suctionOn  = false → IO #10017 = 0  (suction OFF, release)
// =========================================================
bool writeIO(
    rclcpp::Client<WriteSingleIO>::SharedPtr client,
    const rclcpp::Logger& logger,
    uint32_t address,
    int32_t  value,
    int      timeout_ms = 3000)
{
  auto req     = std::make_shared<WriteSingleIO::Request>();
  req->address = address;
  req->value   = value;

  auto fut = client->async_send_request(req);
  if (fut.wait_for(std::chrono::milliseconds(timeout_ms))
      != std::future_status::ready)
  {
    RCLCPP_ERROR(logger,
        "  ❌ WriteSingleIO timeout  address=%u  value=%d", address, value);
    return false;
  }

  auto res = fut.get();
  if (!res->success)
  {
    RCLCPP_ERROR(logger,
        "  ❌ WriteSingleIO failed  address=%u  value=%d  msg=%s",
        address, value, res->message.c_str());
    return false;
  }

  RCLCPP_INFO(logger,
      "  ✔ WriteSingleIO  address=%u  value=%d  OK", address, value);
  return true;
}

bool setSuction(
    rclcpp::Client<WriteSingleIO>::SharedPtr client,
    const rclcpp::Logger& logger,
    bool suctionOn)
{
  RCLCPP_INFO(logger,
      "  🌀 Suction %s  (IO #%u = %d)",
      suctionOn ? "ON" : "OFF",
      IO_SUCTION, suctionOn ? 1 : 0);

  if (!writeIO(client, logger, IO_SUCTION, suctionOn ? 1 : 0))
    return false;

  // Give the vacuum pump / valve time to actuate
  rclcpp::sleep_for(std::chrono::milliseconds(600));
  RCLCPP_INFO(logger, "  ✔ Suction %s done.", suctionOn ? "ON" : "OFF");
  return true;
}

// =========================================================
// Single Cartesian step
// =========================================================
bool cartesianMove(
    MoveGroupInterface& move_group,
    const geometry_msgs::msg::Pose& target,
    const rclcpp::Logger& logger,
    const std::string& name,
    double threshold = 0.95,
    double position_tolerance = 0.05)
{
  auto current = move_group.getCurrentPose().pose;

  RCLCPP_INFO(logger, "[%s] FROM: X=%.4f Y=%.4f Z=%.4f",
              name.c_str(),
              current.position.x, current.position.y, current.position.z);
  RCLCPP_INFO(logger, "[%s]   TO: X=%.4f Y=%.4f Z=%.4f",
              name.c_str(),
              target.position.x, target.position.y, target.position.z);
  RCLCPP_INFO(logger, "[%s] DELTA: dX=%.4f dY=%.4f dZ=%.4f",
              name.c_str(),
              target.position.x - current.position.x,
              target.position.y - current.position.y,
              target.position.z - current.position.z);

  std::vector<geometry_msgs::msg::Pose> waypoints{target};
  moveit_msgs::msg::RobotTrajectory traj;

  double fraction = move_group.computeCartesianPath(
      waypoints, 0.01, 0.0, traj);

  RCLCPP_INFO(logger, "[%s] Coverage: %.1f%% (%zu waypoints)",
              name.c_str(), fraction * 100.0,
              traj.joint_trajectory.points.size());

  if (fraction < threshold)
  {
    RCLCPP_ERROR(logger, "[%s] ❌ Planning failed: %.1f%%",
                 name.c_str(), fraction * 100.0);
    return false;
  }

  MoveGroupInterface::Plan plan;
  plan.trajectory_ = traj;

  auto result = move_group.execute(plan);
  move_group.stop();

  auto after = move_group.getCurrentPose().pose;
  double err = std::sqrt(
      std::pow(after.position.x - target.position.x, 2) +
      std::pow(after.position.y - target.position.y, 2) +
      std::pow(after.position.z - target.position.z, 2));

  RCLCPP_INFO(logger, "[%s] After: X=%.4f Y=%.4f Z=%.4f  err=%.4fm",
              name.c_str(),
              after.position.x, after.position.y, after.position.z, err);

  if (result != moveit::core::MoveItErrorCode::SUCCESS)
  {
    if (err < position_tolerance)
    {
      RCLCPP_WARN(logger,
          "[%s] ⚠ err=%.4fm < %.3fm → SUCCESS (Yaskawa quirk)",
          name.c_str(), err, position_tolerance);
      return true;
    }
    RCLCPP_ERROR(logger, "[%s] ❌ err=%.4fm > %.3fm → FAILED",
                 name.c_str(), err, position_tolerance);
    return false;
  }

  RCLCPP_INFO(logger, "[%s] ✔ Done.", name.c_str());
  return true;
}

// =========================================================
// Incremental move
// =========================================================
bool incrementalMove(
    MoveGroupInterface& move_group,
    const geometry_msgs::msg::Pose& final_target,
    const rclcpp::Logger& logger,
    const std::string& name,
    double step_size = 0.05,
    rclcpp::Client<motoros2_interfaces::srv::StartTrajMode>::SharedPtr traj_client = nullptr)
{
  auto start = move_group.getCurrentPose().pose;

  double dx = final_target.position.x - start.position.x;
  double dy = final_target.position.y - start.position.y;
  double dz = final_target.position.z - start.position.z;
  double total_dist = std::sqrt(dx*dx + dy*dy + dz*dz);

  int steps = std::max(1, (int)std::ceil(total_dist / step_size));

  RCLCPP_INFO(logger, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  RCLCPP_INFO(logger, "[%s] total=%.4fm  steps=%d  step_size=%.4fm",
              name.c_str(), total_dist, steps, step_size);
  RCLCPP_INFO(logger, "[%s] Target: X=%.4f Y=%.4f Z=%.4f",
              name.c_str(),
              final_target.position.x,
              final_target.position.y,
              final_target.position.z);

  for (int i = 1; i <= steps; i++)
  {
    double alpha = (double)i / steps;

    geometry_msgs::msg::Pose step_target = start;
    step_target.position.x = start.position.x + dx * alpha;
    step_target.position.y = start.position.y + dy * alpha;
    step_target.position.z = start.position.z + dz * alpha;

    std::string step_name = name + "-step" +
                            std::to_string(i) + "/" +
                            std::to_string(steps);

    RCLCPP_INFO(logger, "━━━━━ [%s] (%d/%d) ━━━━━",
                name.c_str(), i, steps);

    if (traj_client)
    {
      traj_client->async_send_request(
          std::make_shared<motoros2_interfaces::srv::StartTrajMode::Request>());
      rclcpp::sleep_for(std::chrono::milliseconds(300));
    }

    if (!cartesianMove(move_group, step_target, logger, step_name))
    {
      RCLCPP_ERROR(logger, "[%s] ❌ Failed at step %d/%d. Stopping.",
                   name.c_str(), i, steps);
      return false;
    }

    RCLCPP_INFO(logger, "[%s] ✔ Step %d/%d done.", name.c_str(), i, steps);
  }

  RCLCPP_INFO(logger, "[%s] ✔ All %d steps complete.", name.c_str(), steps);
  RCLCPP_INFO(logger, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  return true;
}

// =========================================================
// MAIN
// =========================================================
int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>(
      "green_pick_homography",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto frame_node = std::make_shared<FrameSubscriber>();
  auto logger = node->get_logger();

  auto declare_safe = [&](const std::string& name, auto val) {
    if (!node->has_parameter(name)) node->declare_parameter(name, val);
  };

  declare_safe("homography_path",
      std::string("/home/karunya/YOLO/homography.txt"));
  declare_safe("step_size", 0.05);
  declare_safe("target_z",  0.20);

  std::string hpath = node->get_parameter("homography_path").as_string();
  double step_size  = node->get_parameter("step_size").as_double();
  double target_z   = node->get_parameter("target_z").as_double();

  RCLCPP_INFO(logger, "════════════════════════════════════════");
  RCLCPP_INFO(logger, "  green_pick_homography  (pick-place)");
  RCLCPP_INFO(logger, "  Detection      : HOUGH circle (+ Hull ellipse fallback)");
  RCLCPP_INFO(logger, "  Hough dp/p1/p2 : %.1f / %.0f / %.2f",
              HOUGH_DP, HOUGH_PARAM1, HOUGH_PARAM2);
  RCLCPP_INFO(logger, "  Hough r-frac   : %.2f ± %.0f%%",
              HOUGH_RADIUS_FRACTION, HOUGH_RADIUS_TOLERANCE * 100.0);
  RCLCPP_INFO(logger, "  homography     : %s",   hpath.c_str());
  RCLCPP_INFO(logger, "  step_size      : %.3fm (%.1fcm)",
              step_size, step_size * 100.0);
  RCLCPP_INFO(logger, "  target_z       : %.4fm (%.1fcm)",
              target_z, target_z * 100.0);
  RCLCPP_INFO(logger, "  lift_height    : %.3fm (%.1fcm)",
              LIFT_HEIGHT, LIFT_HEIGHT * 100.0);
  RCLCPP_INFO(logger, "  retreat_height : %.3fm (%.1fcm)",
              RETREAT_HEIGHT, RETREAT_HEIGHT * 100.0);
  RCLCPP_INFO(logger, "  place_offset   : (+%.3f, +%.3f) m",
              PLACE_OFFSET_X, PLACE_OFFSET_Y);
  RCLCPP_INFO(logger, "  Suction IO     : #%u  (1=ON / 0=OFF)",
              IO_SUCTION);
  RCLCPP_INFO(logger, "  Suction svc    : /yaskawa_hc10/write_single_io");
  RCLCPP_INFO(logger, "────────────────────────────────────────");
  RCLCPP_INFO(logger, "  Pick-place workflow (after SPACE lock):");
  RCLCPP_INFO(logger, "    0. Suction OFF (safe start)");
  RCLCPP_INFO(logger, "    1. XY move to object");
  RCLCPP_INFO(logger, "    2. Z descend to target_z");
  RCLCPP_INFO(logger, "    3. Suction ON  (pick)");
  RCLCPP_INFO(logger, "    4. Z lift +%.1fcm", LIFT_HEIGHT * 100.0);
  RCLCPP_INFO(logger, "    5. XY move to place (+%.1fcm, +%.1fcm)",
              PLACE_OFFSET_X * 100.0, PLACE_OFFSET_Y * 100.0);
  RCLCPP_INFO(logger, "    5b. Z lower 5cm before release");
  RCLCPP_INFO(logger, "    6. Suction OFF (release)");
  RCLCPP_INFO(logger, "    7. Z retreat +%.1fcm", RETREAT_HEIGHT * 100.0);
  RCLCPP_INFO(logger, "    8. Return to home pose (snapshot at lock)");
  RCLCPP_INFO(logger, "════════════════════════════════════════");

  cv::Mat H;
  if (!loadHomography(hpath, H))
  {
    RCLCPP_ERROR(logger, "❌ Failed to load homography: %s", hpath.c_str());
    return 1;
  }
  RCLCPP_INFO(logger, "✔ Homography loaded.");

  // ── Executor ───────────────────────────────────────────
  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.add_node(frame_node);
  std::thread t([&]() { exec.spin(); });

  // ── Wait for camera ────────────────────────────────────
  RCLCPP_INFO(logger, "Waiting for /video_frames...");
  {
    cv::Mat tmp;
    int waited = 0;
    while (rclcpp::ok() && !frame_node->getFrame(tmp))
    {
      rclcpp::sleep_for(std::chrono::milliseconds(200));
      if (++waited % 10 == 0)
        RCLCPP_WARN(logger, "Still waiting for camera... (%ds)", waited/5);
      if (waited > 100) {
        RCLCPP_ERROR(logger, "❌ No camera after 20s.");
        exec.cancel(); t.join();
        return 1;
      }
    }
  }
  RCLCPP_INFO(logger, "✔ Camera ready.");

  // ── Yaskawa trajectory mode ────────────────────────────
  // FIX: correct service name is /yaskawa_hc10/start_traj_mode
  auto traj_client =
      node->create_client<motoros2_interfaces::srv::StartTrajMode>(
          "/yaskawa_hc10/start_traj_mode");
  traj_client->wait_for_service();
  traj_client->async_send_request(
      std::make_shared<motoros2_interfaces::srv::StartTrajMode::Request>());
  RCLCPP_INFO(logger, "✔ Trajectory mode request sent.");

  // ── Suction IO client (WriteSingleIO) ─────────────────
  // FIX: correct service name is /yaskawa_hc10/write_single_io
  auto suction_client =
      node->create_client<WriteSingleIO>("/yaskawa_hc10/write_single_io");
  RCLCPP_INFO(logger, "Waiting for /yaskawa_hc10/write_single_io...");
  suction_client->wait_for_service();
  RCLCPP_INFO(logger, "✔ Suction IO service ready.");

  RCLCPP_INFO(logger, "Waiting 3s for action server...");
  rclcpp::sleep_for(std::chrono::seconds(3));

  // ── MoveIt ─────────────────────────────────────────────
  MoveGroupInterface move_group(node, "manipulator");
  move_group.setMaxVelocityScalingFactor(0.0375);
  move_group.setMaxAccelerationScalingFactor(0.0375);
  move_group.startStateMonitor();

  RCLCPP_INFO(logger, "Waiting 3s for robot state...");
  rclcpp::sleep_for(std::chrono::seconds(3));

  auto init_pose = move_group.getCurrentPose().pose;
  RCLCPP_INFO(logger, "════════════════════════════════════════");
  RCLCPP_INFO(logger, "✔ Robot ready at X=%.4f Y=%.4f Z=%.4f",
              init_pose.position.x,
              init_pose.position.y,
              init_pose.position.z);
  RCLCPP_INFO(logger, "   Z drop will be: %.4fm (%.1fcm)",
              init_pose.position.z - target_z,
              (init_pose.position.z - target_z) * 100.0);

  if (target_z >= init_pose.position.z)
    RCLCPP_WARN(logger,
        "⚠ target_z=%.4f >= current Z=%.4f! Robot will go UP.",
        target_z, init_pose.position.z);

  RCLCPP_INFO(logger, "════════════════════════════════════════");
  RCLCPP_INFO(logger,
      "Waiting for red cylinder. SPACE=lock top-face. ESC=quit.");

  // =========================================================
  // State machine
  // =========================================================
  enum class State { LIVE, FROZEN, DONE };
  State state = State::LIVE;

  cv::Mat      frozen_frame;
  double       target_rx = 0, target_ry = 0;
  TopFaceResult frozen_result;
  int           frame_count = 0;

  geometry_msgs::msg::Pose home_pose;

  while (rclcpp::ok())
  {
    int key = cv::waitKey(30);
    if (key == 27) { RCLCPP_INFO(logger, "ESC. Quitting."); break; }

    // ── LIVE ──────────────────────────────────────────────
    if (state == State::LIVE)
    {
      cv::Mat frame;
      if (!frame_node->getFrame(frame)) continue;
      frame_count++;

      TopFaceResult r;
      bool found = detectCylinderTop(frame, r);

      double rx = 0, ry = 0;
      if (found)
      {
        std::vector<cv::Point2f> src{{(float)r.px, (float)r.py}};
        std::vector<cv::Point2f> dst;
        cv::perspectiveTransform(src, dst, H);
        rx = dst[0].x;
        ry = dst[0].y;
        drawTopFaceOverlay(frame, r, rx, ry);
      }

      cv::line(frame, {320-20, 240}, {320+20, 240}, {255,255,255}, 1);
      cv::line(frame, {320, 240-20}, {320, 240+20}, {255,255,255}, 1);

      cv::putText(frame,
          found ? (r.method == TopFaceResult::Method::HOUGH
                    ? "[H] HOUGH circle  - SPACE to lock"
                    : r.method == TopFaceResult::Method::HULL
                      ? "[E] HULL ellipse  - SPACE to lock"
                      : "[B] BBOX fallback - SPACE to lock")
                : "Searching for red cylinder...",
          {10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
          found ? (r.method == TopFaceResult::Method::HOUGH
                    ? cv::Scalar(0,220,0)
                    : r.method == TopFaceResult::Method::HULL
                      ? cv::Scalar(0,165,255)
                      : cv::Scalar(0,0,255))
                : cv::Scalar(0,100,255), 1);

      cv::putText(frame, "MODE: LIVE  ESC=quit",
          {10, 45}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {255,255,0}, 1);

      auto live = move_group.getCurrentPose().pose;
      cv::putText(frame,
          "RobotZ=" + std::to_string(live.position.z).substr(0,6) +
          "m  targetZ=" + std::to_string(target_z).substr(0,6) + "m",
          {10, 70}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {200,200,255}, 1);

      if (found && r.method != TopFaceResult::Method::BBOX)
        cv::putText(frame,
            "r=" + std::to_string(static_cast<int>(r.radius)) +
            "px  axes=" + std::to_string(static_cast<int>(r.axis_major)) +
            "x" + std::to_string(static_cast<int>(r.axis_minor)) + "px",
            {10, 95}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {200,255,200}, 1);

      cv::imshow("Green Cylinder Tracker", frame);

      if (key == 32 && found)
      {
        home_pose      = move_group.getCurrentPose().pose;
        frozen_frame   = frame.clone();
        target_rx      = rx;
        target_ry      = ry;
        frozen_result  = r;
        state          = State::FROZEN;

        const char* mname =
            (r.method == TopFaceResult::Method::HOUGH) ? "HOUGH circle" :
            (r.method == TopFaceResult::Method::HULL)  ? "HULL ellipse" :
                                                          "BBOX fallback";

        RCLCPP_INFO(logger, "════════════════════════════════════════");
        RCLCPP_INFO(logger, "🔒 FRAME LOCKED at frame #%d", frame_count);
        RCLCPP_INFO(logger, "   Detection method  : %s", mname);
        RCLCPP_INFO(logger, "   Centroid pixel    : (%d, %d)", r.px, r.py);
        RCLCPP_INFO(logger, "   Circle radius     : %.1f px", r.radius);
        RCLCPP_INFO(logger, "   Ellipse axes      : %.1f x %.1f px",
                    r.axis_major, r.axis_minor);
        RCLCPP_INFO(logger, "   Robot XY          : (%.4f, %.4f)",
                    target_rx, target_ry);
        RCLCPP_INFO(logger, "   Target Z          : %.4fm", target_z);
        RCLCPP_INFO(logger, "   Home pose snapshot: X=%.4f Y=%.4f Z=%.4f",
                    home_pose.position.x,
                    home_pose.position.y,
                    home_pose.position.z);
        RCLCPP_INFO(logger, "   Press SPACE to begin pick-place.");
        RCLCPP_INFO(logger, "════════════════════════════════════════");

        cv::Mat disp = frozen_frame.clone();
        cv::putText(disp, "LOCKED. SPACE = start pick-place.",
            {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,165,255}, 2);
        cv::putText(disp,
            std::string(mname) +
            "  Top:(" + std::to_string(r.px) + "," + std::to_string(r.py) +
            ")  XY:(" + std::to_string(target_rx).substr(0,6) +
            "," + std::to_string(target_ry).substr(0,6) +
            ")  Z->" + std::to_string(target_z).substr(0,6) + "m",
            {10,50}, cv::FONT_HERSHEY_SIMPLEX, 0.42, {0,165,255}, 1);
        cv::imshow("Green Cylinder Tracker", disp);
      }
      else if (key == 32 && !found)
      {
        RCLCPP_WARN(logger, "⚠ SPACE pressed but no cylinder detected.");
      }
    }

    // ── FROZEN ────────────────────────────────────────────
    else if (state == State::FROZEN)
    {
      cv::Mat disp = frozen_frame.clone();
      cv::putText(disp, "LOCKED. SPACE = start pick-place. ESC = quit.",
          {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,165,255}, 2);
      cv::putText(disp,
          "Top:(" + std::to_string(frozen_result.px) + "," +
          std::to_string(frozen_result.py) +
          ")  XY:(" + std::to_string(target_rx).substr(0,6) +
          "," + std::to_string(target_ry).substr(0,6) +
          ")  r=" + std::to_string(static_cast<int>(frozen_result.radius)) +
          "px  Z->" + std::to_string(target_z).substr(0,6) + "m",
          {10,50}, cv::FONT_HERSHEY_SIMPLEX, 0.42, {0,165,255}, 1);
      cv::imshow("Green Cylinder Tracker", disp);

      if (key != 32) continue;

      // ── Status overlay helper ──────────────────────────
      auto show_status = [&](const std::string& msg, cv::Scalar color) {
        cv::Mat d = frozen_frame.clone();
        cv::putText(d, msg, {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        auto p = move_group.getCurrentPose().pose;
        cv::putText(d,
            "RobotZ=" + std::to_string(p.position.z).substr(0,6) +
            "m  targetZ=" + std::to_string(target_z).substr(0,6) + "m",
            {10,50}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {200,200,255}, 1);
        cv::imshow("Green Cylinder Tracker", d);
        cv::waitKey(1);
      };

      auto poseAt = [&](double x, double y, double z) {
        auto p = move_group.getCurrentPose().pose;
        p.position.x = x;
        p.position.y = y;
        p.position.z = z;
        return p;
      };

      RCLCPP_INFO(logger, "════════════════════════════════════════");
      RCLCPP_INFO(logger, "🚀 PICK-PLACE SEQUENCE START");
      RCLCPP_INFO(logger, "   Pick XY  : (%.4f, %.4f)", target_rx, target_ry);
      RCLCPP_INFO(logger, "   Pick Z   : %.4fm", target_z);
      RCLCPP_INFO(logger, "   Place XY : (%.4f, %.4f)",
                  target_rx + PLACE_OFFSET_X,
                  target_ry + PLACE_OFFSET_Y);
      RCLCPP_INFO(logger, "   step_size: %.3fm", step_size);
      RCLCPP_INFO(logger, "════════════════════════════════════════");

      bool seq_ok = true;

      // ─────────────────────────────────────────────────────
      // STEP 0 — Suction OFF before approaching
      // ─────────────────────────────────────────────────────
      show_status("STEP 0: Suction OFF (safe start)...", {0,165,255});
      RCLCPP_INFO(logger, "──── STEP 0: Suction OFF ────");
      if (!setSuction(suction_client, logger, /*on=*/false)) {
        show_status("STEP 0 FAILED: suction OFF error.", {0,0,255});
        seq_ok = false; goto seq_end;
      }
      RCLCPP_INFO(logger, "✔ Suction off.");

      // ─────────────────────────────────────────────────────
      // STEP 1 — XY move to above object
      // ─────────────────────────────────────────────────────
      {
        show_status("STEP 1: Moving XY to object...", {0,165,255});
        RCLCPP_INFO(logger, "──── STEP 1: XY move ────");
        auto cur    = move_group.getCurrentPose().pose;
        auto xy_tgt = poseAt(target_rx, target_ry, cur.position.z);
        if (!incrementalMove(move_group, xy_tgt, logger,
                             "XY", step_size, traj_client)) {
          show_status("STEP 1 FAILED. SPACE to retry.", {0,0,255});
          seq_ok = false; goto seq_end;
        }
        RCLCPP_INFO(logger, "✔ Step 1 XY complete.");
      }

      // ─────────────────────────────────────────────────────
      // STEP 2 — Z descend to object
      // ─────────────────────────────────────────────────────
      {
        show_status("STEP 2: Descending to object...", {0,165,255});
        RCLCPP_INFO(logger, "──── STEP 2: Z descend to %.4fm ────", target_z);
        auto after_xy = move_group.getCurrentPose().pose;
        RCLCPP_INFO(logger,
            "   From Z=%.4f → To Z=%.4f  (drop=%.4fm / %.1fcm)",
            after_xy.position.z, target_z,
            after_xy.position.z - target_z,
            (after_xy.position.z - target_z) * 100.0);
        auto z_tgt = poseAt(after_xy.position.x, after_xy.position.y, target_z);
        if (!incrementalMove(move_group, z_tgt, logger,
                             "Z_down", step_size, traj_client)) {
          show_status("STEP 2 FAILED. SPACE to retry.", {0,0,255});
          seq_ok = false; goto seq_end;
        }
        RCLCPP_INFO(logger, "✔ Step 2 Z descend complete.");
      }

      // ─────────────────────────────────────────────────────
      // STEP 3 — Suction ON (pick)
      // ─────────────────────────────────────────────────────
      show_status("STEP 3: Suction ON (pick)...", {0,165,255});
      RCLCPP_INFO(logger, "──── STEP 3: Suction ON ────");
      if (!setSuction(suction_client, logger, /*on=*/true)) {
        show_status("STEP 3 FAILED: suction ON error.", {0,0,255});
        seq_ok = false; goto seq_end;
      }
      RCLCPP_INFO(logger, "✔ Suction on — object picked.");

      // ─────────────────────────────────────────────────────
      // STEP 4 — Lift
      // ─────────────────────────────────────────────────────
      {
        show_status("STEP 4: Lifting object...", {0,165,255});
        RCLCPP_INFO(logger, "──── STEP 4: Z lift +%.1fcm ────",
                    LIFT_HEIGHT * 100.0);
        auto at_pick  = move_group.getCurrentPose().pose;
        auto lift_tgt = poseAt(at_pick.position.x,
                               at_pick.position.y,
                               at_pick.position.z + LIFT_HEIGHT);
        if (!incrementalMove(move_group, lift_tgt, logger,
                             "Z_lift", step_size, traj_client)) {
          show_status("STEP 4 FAILED. SPACE to retry.", {0,0,255});
          seq_ok = false; goto seq_end;
        }
        RCLCPP_INFO(logger, "✔ Step 4 lift complete.");
      }

      // ─────────────────────────────────────────────────────
      // STEP 5 — XY move to place location
      // ─────────────────────────────────────────────────────
      {
        double place_x = target_rx + PLACE_OFFSET_X;
        double place_y = target_ry + PLACE_OFFSET_Y;
        show_status("STEP 5: Moving to place location...", {0,165,255});
        RCLCPP_INFO(logger,
            "──── STEP 5: XY move to place (%.4f, %.4f) ────",
            place_x, place_y);
        auto cur       = move_group.getCurrentPose().pose;
        auto place_tgt = poseAt(place_x, place_y, cur.position.z);
        if (!incrementalMove(move_group, place_tgt, logger,
                             "XY_place", step_size, traj_client)) {
          show_status("STEP 5 FAILED. SPACE to retry.", {0,0,255});
          seq_ok = false; goto seq_end;
        }
        RCLCPP_INFO(logger, "✔ Step 5 XY place complete.");
      }

      // ─────────────────────────────────────────────────────
      // STEP 5b — Lower 5 cm before releasing
      // ─────────────────────────────────────────────────────
      {
        static constexpr double PLACE_LOWER = 0.05;
        show_status("STEP 5b: Lowering 5cm before release...", {0,165,255});
        RCLCPP_INFO(logger, "──── STEP 5b: Z lower -%.1fcm at place ────",
                    PLACE_LOWER * 100.0);
        auto at_place  = move_group.getCurrentPose().pose;
        auto lower_tgt = poseAt(at_place.position.x,
                                at_place.position.y,
                                at_place.position.z - PLACE_LOWER);
        RCLCPP_INFO(logger, "   From Z=%.4f → To Z=%.4f",
                    at_place.position.z, lower_tgt.position.z);
        if (!incrementalMove(move_group, lower_tgt, logger,
                             "Z_place_lower", step_size, traj_client)) {
          show_status("STEP 5b FAILED. SPACE to retry.", {0,0,255});
          seq_ok = false; goto seq_end;
        }
        RCLCPP_INFO(logger, "✔ Step 5b lower complete.");
      }

      // ─────────────────────────────────────────────────────
      // STEP 6 — Suction OFF (release)
      // ─────────────────────────────────────────────────────
      show_status("STEP 6: Suction OFF (release)...", {0,165,255});
      RCLCPP_INFO(logger, "──── STEP 6: Suction OFF ────");
      if (!setSuction(suction_client, logger, /*on=*/false)) {
        show_status("STEP 6 FAILED: suction OFF error.", {0,0,255});
        seq_ok = false; goto seq_end;
      }
      RCLCPP_INFO(logger, "✔ Suction off — object released.");

      // ─────────────────────────────────────────────────────
      // STEP 7 — Retreat upward
      // ─────────────────────────────────────────────────────
      {
        show_status("STEP 7: Retreating upward...", {0,165,255});
        RCLCPP_INFO(logger, "──── STEP 7: Z retreat +%.1fcm ────",
                    RETREAT_HEIGHT * 100.0);
        auto at_place    = move_group.getCurrentPose().pose;
        auto retreat_tgt = poseAt(at_place.position.x,
                                  at_place.position.y,
                                  at_place.position.z + RETREAT_HEIGHT);
        if (!incrementalMove(move_group, retreat_tgt, logger,
                             "Z_retreat", step_size, traj_client)) {
          show_status("STEP 7 FAILED. SPACE to retry.", {0,0,255});
          seq_ok = false; goto seq_end;
        }
        RCLCPP_INFO(logger, "✔ Step 7 retreat complete.");
      }

      // ─────────────────────────────────────────────────────
      // STEP 8 — Return to home pose
      // ─────────────────────────────────────────────────────
      {
        show_status("STEP 8: Returning to home pose...", {0,165,255});
        RCLCPP_INFO(logger, "──── STEP 8: Return home ────");
        RCLCPP_INFO(logger, "   Home: X=%.4f Y=%.4f Z=%.4f",
                    home_pose.position.x,
                    home_pose.position.y,
                    home_pose.position.z);

        auto cur       = move_group.getCurrentPose().pose;
        auto raise_tgt = poseAt(cur.position.x,
                                cur.position.y,
                                home_pose.position.z);
        RCLCPP_INFO(logger, "   8a: raise Z from %.4f to %.4f",
                    cur.position.z, home_pose.position.z);
        if (!incrementalMove(move_group, raise_tgt, logger,
                             "Home_Z", step_size, traj_client)) {
          show_status("STEP 8a (Z raise) FAILED. SPACE to retry.", {0,0,255});
          seq_ok = false; goto seq_end;
        }

        auto home_tgt = poseAt(home_pose.position.x,
                               home_pose.position.y,
                               home_pose.position.z);
        RCLCPP_INFO(logger, "   8b: XY to (%.4f, %.4f)",
                    home_pose.position.x, home_pose.position.y);
        if (!incrementalMove(move_group, home_tgt, logger,
                             "Home_XY", step_size, traj_client)) {
          show_status("STEP 8b (XY home) FAILED. SPACE to retry.", {0,0,255});
          seq_ok = false; goto seq_end;
        }
        RCLCPP_INFO(logger, "✔ Step 8 return home complete.");
      }

      seq_end:
      {
        auto final_p = move_group.getCurrentPose().pose;
        RCLCPP_INFO(logger, "════════════════════════════════════════");
        if (seq_ok)
          RCLCPP_INFO(logger, "✔ PICK-PLACE SEQUENCE COMPLETE.");
        else
          RCLCPP_ERROR(logger, "❌ SEQUENCE ENDED WITH ERRORS.");
        RCLCPP_INFO(logger, "   Final: X=%.4f Y=%.4f Z=%.4f",
                    final_p.position.x,
                    final_p.position.y,
                    final_p.position.z);
        RCLCPP_INFO(logger, "════════════════════════════════════════");

        state = State::DONE;
        show_status(seq_ok
            ? "DONE. Pick-place complete. ESC to quit."
            : "SEQUENCE FAILED. ESC to quit.",
            seq_ok ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255));
      }
    }

    // ── DONE ──────────────────────────────────────────────
    else if (state == State::DONE)
    {
      auto final_p = move_group.getCurrentPose().pose;
      cv::Mat disp = frozen_frame.clone();
      cv::putText(disp, "DONE. ESC to quit.",
          {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);
      cv::putText(disp,
          "Final Z=" + std::to_string(final_p.position.z).substr(0,6) +
          "m  home=" + std::to_string(home_pose.position.z).substr(0,6) + "m",
          {10,50}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,255,0}, 1);
      cv::imshow("Green Cylinder Tracker", disp);
    }
  }

  cv::destroyAllWindows();
  exec.cancel();
  t.join();
  rclcpp::shutdown();
  return 0;
}
