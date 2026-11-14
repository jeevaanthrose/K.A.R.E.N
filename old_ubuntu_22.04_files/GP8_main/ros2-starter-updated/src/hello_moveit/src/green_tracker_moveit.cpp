#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <motoros2_interfaces/srv/start_traj_mode.hpp>
#include <motoros2_interfaces/srv/write_single_io.hpp>   // ← correct service

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <thread>
#include <mutex>
#include <string>
#include <cmath>

using moveit::planning_interface::MoveGroupInterface;
using WriteSingleIO = motoros2_interfaces::srv::WriteSingleIO;  // ← correct type

// =========================================================
// USER SETTINGS
// =========================================================

// Minimum contour area (px²) to be considered a valid cylinder
static constexpr double MIN_CONTOUR_AREA = 100.0;

// ── Hough Circle settings ─────────────────────────────────
// dp: inverse resolution ratio (1.0 = full res; 1.5 = slightly downsampled)
//     Lower = more accurate but slower. 1.0–1.5 is fine for a small ROI.
static constexpr double HOUGH_DP           = 1.2;

// Minimum distance between detected circle centres (px).
// Set large enough to prevent duplicate detections of the same cylinder.
static constexpr double HOUGH_MIN_DIST     = 50.0;

// Canny high threshold fed to HoughCircles internally.
// Lower = more edge fragments found (sensitive); higher = fewer, cleaner.
static constexpr double HOUGH_PARAM1       = 100.0;

// Accumulator threshold: higher = only strong, well-defined circles pass.
// If Hough misses the cylinder top, lower this (e.g. 0.7 → 0.5).
// If it finds false circles, raise it (e.g. 0.7 → 0.85).
static constexpr double HOUGH_PARAM2       = 0.75;   // used with ALT method (0–1)

// Hough radius search is auto-derived from the bounding-box width, but
// these fractions let you clamp the search range.
// Expected top-circle radius ≈ bbox.width * HOUGH_RADIUS_FRACTION.
// The search window is [fraction * (1-tolerance) … fraction * (1+tolerance)].
static constexpr double HOUGH_RADIUS_FRACTION  = 0.45;  // top-face r ≈ 45% of bbox w
static constexpr double HOUGH_RADIUS_TOLERANCE = 0.40;  // ±40% around that estimate

// Confidence gate: Hough centre must lie within this fraction of the
// bounding-box width/height from the bbox centre to be accepted.
static constexpr double HOUGH_CENTRE_GATE  = 0.55;

// IO addresses (Universal/General Outputs — 10010 and up are writable)
static constexpr uint32_t IO_GRIPPER_CLOSE = 10011;  // write 1 = close
static constexpr uint32_t IO_GRIPPER_OPEN  = 10012;  // write 1 = open

// How far to lift after grasping (metres)
static constexpr double LIFT_HEIGHT = 0.08;     // 8 cm

// How far to retreat upward after placing (metres)
static constexpr double RETREAT_HEIGHT = 0.06;  // 6 cm

// Place offset from pick XY (metres) — nearby drop location
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
// TopFaceResult — carries all outputs of detectCylinderTop
// =========================================================
struct TopFaceResult
{
  int    px = 0, py = 0;          // pixel centroid of top face
  double radius   = 0.0;          // circle radius (px) — Hough or derived
  double axis_major = 0.0;        // ellipse half-major axis (px) — Hull path
  double axis_minor = 0.0;        // ellipse half-minor axis (px) — Hull path
  double angle_deg  = 0.0;        // ellipse orientation (deg)

  // Which detection path was used (for HUD / logging)
  enum class Method { HOUGH, HULL, BBOX } method = Method::BBOX;
};

// =========================================================
// Cylinder top-face detection — Hough Circle + Hull fallback
//
// Algorithm
// ─────────
//  Stage A — HOUGH (preferred, nearly-overhead camera)
//   1. Build red HSV mask + morphological open.
//   2. Find largest contour; reject if area < MIN_CONTOUR_AREA.
//   3. Isolate a padded ROI around the bounding box.
//   4. Gaussian-blur the grayscale ROI to suppress noise.
//   5. HoughCircles (HOUGH_GRADIENT_ALT) with radius range auto-derived
//      from the bounding-box width.
//   6. Confidence gate: accept the best circle only if its centre is
//      within HOUGH_CENTRE_GATE of the bbox centre (rejects strays).
//
//  Stage B — CONVEX HULL + fitEllipse (fallback)
//   7. Compute convexHull of the largest contour.
//   8. fitEllipse on the hull points → cleaner than raw contour points.
//   9. Use ellipse centre as the top-face centroid.
//
//  Stage C — Bounding-box last resort
//  10. bbox top-centre if hull has < 5 points.
// =========================================================
bool detectCylinderTop(
    const cv::Mat& frame,
    TopFaceResult& result)
{
  // ── 1. Red HSV mask ──────────────────────────────────────
  cv::Mat hsv, mask;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  cv::Mat mask1, mask2;
  cv::inRange(hsv, cv::Scalar(0,   100, 100), cv::Scalar(10,  255, 255), mask1);
  cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), mask2);
  cv::bitwise_or(mask1, mask2, mask);
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                   cv::Mat::ones(5, 5, CV_8U));

  // ── 2. Largest contour ───────────────────────────────────
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours,
                   cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return false;

  auto best_it = std::max_element(contours.begin(), contours.end(),
      [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return cv::contourArea(a) < cv::contourArea(b);
      });
  if (cv::contourArea(*best_it) < MIN_CONTOUR_AREA) return false;
  const auto& contour = *best_it;

  cv::Rect bbox = cv::boundingRect(contour);

  // ── 3. Padded ROI for Hough ──────────────────────────────
  //  A small pad prevents Hough from missing circles that touch the ROI edge.
  const int pad    = 10;
  cv::Rect roi_rect(
      std::max(0, bbox.x - pad),
      std::max(0, bbox.y - pad),
      std::min(frame.cols - std::max(0, bbox.x - pad), bbox.width  + 2*pad),
      std::min(frame.rows - std::max(0, bbox.y - pad), bbox.height + 2*pad));

  // ── 4. Prepare Hough input ───────────────────────────────
  //  Work on the masked ROI in grayscale so only red pixels contribute.
  cv::Mat roi_mask  = mask(roi_rect);
  cv::Mat roi_gray;
  cv::cvtColor(frame(roi_rect), roi_gray, cv::COLOR_BGR2GRAY);
  roi_gray.setTo(0, ~roi_mask);               // zero out non-red pixels
  cv::GaussianBlur(roi_gray, roi_gray, {9,9}, 2.0);

  // ── 5. Hough Circle detection ────────────────────────────
  double expected_r = bbox.width * HOUGH_RADIUS_FRACTION;
  int    r_min      = std::max(5, static_cast<int>(expected_r * (1.0 - HOUGH_RADIUS_TOLERANCE)));
  int    r_max      = static_cast<int>(expected_r * (1.0 + HOUGH_RADIUS_TOLERANCE));

  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(roi_gray, circles,
                   cv::HOUGH_GRADIENT_ALT,
                   HOUGH_DP,
                   HOUGH_MIN_DIST,
                   HOUGH_PARAM1,
                   HOUGH_PARAM2,
                   r_min, r_max);

  // ── 6. Confidence gate ───────────────────────────────────
  //  Accept the first circle whose centre is near the bbox centre in ROI space.
  double bbox_cx_roi = (bbox.x - roi_rect.x) + bbox.width  / 2.0;
  double bbox_cy_roi = (bbox.y - roi_rect.y) + bbox.height / 2.0;
  double gate_x      = bbox.width  * HOUGH_CENTRE_GATE;
  double gate_y      = bbox.height * HOUGH_CENTRE_GATE;

  for (const auto& c : circles)
  {
    double cx_roi = c[0],  cy_roi = c[1],  r = c[2];
    if (std::abs(cx_roi - bbox_cx_roi) < gate_x &&
        std::abs(cy_roi - bbox_cy_roi) < gate_y)
    {
      // Convert centre back to full-frame coordinates
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

  // ── 7–9. Hull fallback ───────────────────────────────────
  std::vector<cv::Point> hull;
  cv::convexHull(contour, hull);

  if (hull.size() >= 5)
  {
    cv::RotatedRect ell = cv::fitEllipse(hull);
    result.px         = static_cast<int>(ell.center.x);
    result.py         = static_cast<int>(ell.center.y);
    result.radius     = (ell.size.width + ell.size.height) / 4.0;  // mean
    result.axis_major = ell.size.width  / 2.0;
    result.axis_minor = ell.size.height / 2.0;
    result.angle_deg  = ell.angle;
    result.method     = TopFaceResult::Method::HULL;
    return true;
  }

  // ── 10. Last resort ──────────────────────────────────────
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
// Draw top-face overlay from TopFaceResult
// =========================================================
void drawTopFaceOverlay(cv::Mat& frame, const TopFaceResult& r,
                        double rx, double ry)
{
  // Colour codes: Hough=green, Hull=orange, Bbox=red
  cv::Scalar colour =
      (r.method == TopFaceResult::Method::HOUGH) ? cv::Scalar(0, 220,  0)  :
      (r.method == TopFaceResult::Method::HULL)  ? cv::Scalar(0, 165, 255) :
                                                    cv::Scalar(0,   0, 255);

  if (r.method == TopFaceResult::Method::HOUGH)
  {
    // Perfect circle from Hough
    cv::circle(frame, {r.px, r.py},
               static_cast<int>(r.radius), colour, 2);

    // Cross-hair at centre
    int cr = static_cast<int>(r.radius);
    cv::line(frame, {r.px - cr, r.py}, {r.px + cr, r.py}, colour, 1);
    cv::line(frame, {r.px, r.py - cr}, {r.px, r.py + cr}, colour, 1);
  }
  else
  {
    // Hull ellipse (or bbox fallback)
    if (r.axis_major > 1.0)
      cv::ellipse(frame,
                  cv::Point(r.px, r.py),
                  cv::Size(static_cast<int>(r.axis_major),
                           static_cast<int>(r.axis_minor)),
                  r.angle_deg, 0, 360, colour, 2);

    // Principal-axis line for hull ellipse
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

  // Centroid dot
  cv::circle(frame, {r.px, r.py}, 5, colour, -1);
  cv::circle(frame, {r.px, r.py}, 16, colour, 1);

  // Method tag
  const char* tag =
      (r.method == TopFaceResult::Method::HOUGH) ? "[H]" :
      (r.method == TopFaceResult::Method::HULL)  ? "[E]" : "[B]";

  std::string label =
      std::string(tag) +
      " Top:(" + std::to_string(r.px) + "," + std::to_string(r.py) + ")"
      "  R:(" + std::to_string(rx).substr(0,6) +
      "," + std::to_string(ry).substr(0,6) + ")"
      "  r=" + std::to_string(static_cast<int>(r.radius)) + "px";

  cv::putText(frame, label,
              {r.px + 20, r.py - 10},
              cv::FONT_HERSHEY_SIMPLEX, 0.42, colour, 1);
}

// =========================================================
// Gripper control via /yaskawa/write_single_io
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
        "  ❌ WriteSingleIO timeout  address=%u  value=%d",
        address, value);
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
      "  ✔ WriteSingleIO  address=%u  value=%d  OK",
      address, value);
  return true;
}

bool setGripper(
    rclcpp::Client<WriteSingleIO>::SharedPtr client,
    const rclcpp::Logger& logger,
    bool close)
{
  const char* action = close ? "CLOSE" : "OPEN";
  RCLCPP_INFO(logger,
      "  🤏 Gripper %s  (IO %u=%d  IO %u=%d)",
      action,
      IO_GRIPPER_CLOSE, close ? 1 : 0,
      IO_GRIPPER_OPEN,  close ? 0 : 1);

  uint32_t on_addr  = close ? IO_GRIPPER_CLOSE : IO_GRIPPER_OPEN;
  uint32_t off_addr = close ? IO_GRIPPER_OPEN  : IO_GRIPPER_CLOSE;

  if (!writeIO(client, logger, on_addr, 1)) return false;
  if (!writeIO(client, logger, off_addr, 0)) return false;

  rclcpp::sleep_for(std::chrono::milliseconds(600));
  RCLCPP_INFO(logger, "  ✔ Gripper %s done.", action);
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
// Incremental move — breaks large moves into small chunks
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
  declare_safe("target_z",  0.18);

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
  RCLCPP_INFO(logger, "  Gripper IO     : CLOSE=#%u  OPEN=#%u",
              IO_GRIPPER_CLOSE, IO_GRIPPER_OPEN);
  RCLCPP_INFO(logger, "  Gripper svc    : /yaskawa/write_single_io");
  RCLCPP_INFO(logger, "────────────────────────────────────────");
  RCLCPP_INFO(logger, "  Pick-place workflow (after SPACE lock):");
  RCLCPP_INFO(logger, "    0. Gripper OPEN");
  RCLCPP_INFO(logger, "    1. XY move to object");
  RCLCPP_INFO(logger, "    2. Z descend to target_z");
  RCLCPP_INFO(logger, "    3. Gripper CLOSE  (grasp)");
  RCLCPP_INFO(logger, "    4. Z lift +%.1fcm", LIFT_HEIGHT * 100.0);
  RCLCPP_INFO(logger, "    5. XY move to place (+%.1fcm, +%.1fcm)",
              PLACE_OFFSET_X * 100.0, PLACE_OFFSET_Y * 100.0);
  RCLCPP_INFO(logger, "    6. Gripper OPEN   (release)");
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
  auto traj_client =
      node->create_client<motoros2_interfaces::srv::StartTrajMode>(
          "yaskawa/start_traj_mode");
  traj_client->wait_for_service();
  traj_client->async_send_request(
      std::make_shared<motoros2_interfaces::srv::StartTrajMode::Request>());
  RCLCPP_INFO(logger, "✔ Trajectory mode request sent.");

  // ── Gripper IO client (WriteSingleIO) ─────────────────
  auto gripper_client =
      node->create_client<WriteSingleIO>("/yaskawa/write_single_io");
  RCLCPP_INFO(logger, "Waiting for /yaskawa/write_single_io...");
  gripper_client->wait_for_service();
  RCLCPP_INFO(logger, "✔ Gripper IO service ready.");

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

  // Robot pose snapshot taken at frame-lock — used as home to return to
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

      // Crosshair
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

      // ── Build pose at given XYZ (preserves orientation) ──
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
      // STEP 0 — Open gripper before approaching
      // ─────────────────────────────────────────────────────
      show_status("STEP 0: Opening gripper...", {0,165,255});
      RCLCPP_INFO(logger, "──── STEP 0: Gripper OPEN ────");
      if (!setGripper(gripper_client, logger, /*close=*/false)) {
        show_status("STEP 0 FAILED: gripper open error.", {0,0,255});
        seq_ok = false; goto seq_end;
      }
      RCLCPP_INFO(logger, "✔ Gripper open.");

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
      // STEP 3 — Close gripper (grasp)
      // ─────────────────────────────────────────────────────
      show_status("STEP 3: Closing gripper (grasp)...", {0,165,255});
      RCLCPP_INFO(logger, "──── STEP 3: Gripper CLOSE ────");
      if (!setGripper(gripper_client, logger, /*close=*/true)) {
        show_status("STEP 3 FAILED: gripper close error.", {0,0,255});
        seq_ok = false; goto seq_end;
      }
      RCLCPP_INFO(logger, "✔ Gripper closed — object grasped.");

      // ─────────────────────────────────────────────────────
      // STEP 4 — Lift (Z up by LIFT_HEIGHT)
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
      // STEP 5b — Lower 5 cm before releasing at place
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
      // STEP 6 — Open gripper (release)
      // ─────────────────────────────────────────────────────
      show_status("STEP 6: Opening gripper (release)...", {0,165,255});
      RCLCPP_INFO(logger, "──── STEP 6: Gripper OPEN ────");
      if (!setGripper(gripper_client, logger, /*close=*/false)) {
        show_status("STEP 6 FAILED: gripper open error.", {0,0,255});
        seq_ok = false; goto seq_end;
      }
      RCLCPP_INFO(logger, "✔ Gripper open — object released.");

      // ─────────────────────────────────────────────────────
      // STEP 7 — Retreat upward (Z up by RETREAT_HEIGHT)
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
      // STEP 8 — Return to home pose (snapshotted at lock)
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
