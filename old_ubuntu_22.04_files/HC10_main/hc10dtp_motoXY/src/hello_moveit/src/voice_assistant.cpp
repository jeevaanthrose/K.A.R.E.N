// voice_assistant.cpp
// ═══════════════════════════════════════════════════════════════════════════
// ROS2 / MoveIt pick-and-place node driven by two ROS2 parameters:
//   color  (string) — e.g. "green", "red", "blue", "yellow", "orange",
//                          "purple", "white", "black", "brown"
//   shape  (string) — "cylinder"  or  "cuboid"
//
// Build alongside green_tracker_moveit.cpp; add to CMakeLists.txt:
//   add_executable(voice_assistant src/voice_assistant.cpp)
//   ament_target_dependencies(voice_assistant rclcpp moveit_ros_planning_interface
//     motoros2_interfaces OpenCV cv_bridge sensor_msgs)
//
// Launch (done automatically by voice_commander.py):
//   ros2 run hello_moveit voice_assistant \
//       --ros-args -p color:=green -p shape:=cylinder
// ═══════════════════════════════════════════════════════════════════════════

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
#include <vector>
#include <cmath>
#include <algorithm>
#include <cctype>

using moveit::planning_interface::MoveGroupInterface;
using WriteSingleIO = motoros2_interfaces::srv::WriteSingleIO;

// ─────────────────────────────────────────────────────────────────────────────
// USER SETTINGS
// ─────────────────────────────────────────────────────────────────────────────

static constexpr double MIN_CONTOUR_AREA  = 100.0;

// Hough circle (cylinder detection)
static constexpr double HOUGH_DP                = 1.2;
static constexpr double HOUGH_MIN_DIST          = 50.0;
static constexpr double HOUGH_PARAM1            = 100.0;
static constexpr double HOUGH_PARAM2            = 0.75;
static constexpr double HOUGH_RADIUS_FRACTION   = 0.45;
static constexpr double HOUGH_RADIUS_TOLERANCE  = 0.40;
static constexpr double HOUGH_CENTRE_GATE       = 0.55;

// Cuboid detection
// Aspect ratio (w/h) of the bounding rect that qualifies as cuboid top-face
static constexpr double CUBOID_AR_MIN     = 0.55;   // almost square
static constexpr double CUBOID_AR_MAX     = 2.50;   // long rectangles OK
// Circularity below this → definitely not a circle → treat as cuboid
static constexpr double CUBOID_CIRC_MAX   = 0.82;

// Cylinder rejection guard: prevents a rectangular colored blob from being
// accepted by the cylinder HULL/BBOX fallbacks.
static constexpr double RECTANGULAR_EXTENT_MIN = 0.82;
static constexpr double RECTANGULAR_CIRC_MAX   = 0.86;

// Suction gripper
static constexpr uint32_t IO_SUCTION      = 10017;

// Motion constants (metres)
static constexpr double LIFT_HEIGHT       = 0.08;
static constexpr double RETREAT_HEIGHT    = 0.06;
static constexpr double PLACE_OFFSET_X   =  0.08;
static constexpr double PLACE_OFFSET_Y   =  0.08;
static constexpr double DEFAULT_PICK_Z   =  0.10;
static constexpr double DEFAULT_DROP_Z   =  0.08;
static constexpr double DEFAULT_MIN_Z    =  0.08;

// ─────────────────────────────────────────────────────────────────────────────
// Color → HSV range(s)
//   Each entry holds one or two {lower, upper} pairs (red wraps around 0/180).
// ─────────────────────────────────────────────────────────────────────────────
struct HsvRange { cv::Scalar lo, hi; };

struct ColorDef {
  std::string   name;
  std::vector<HsvRange> ranges;
};

// Returns true and fills `mask` for the requested color name.
// Mask is CV_8U, same size as `hsv`.
bool colorMask(const cv::Mat& hsv, const std::string& color, cv::Mat& mask)
{
  // clang-format off
  static const std::vector<ColorDef> TABLE = {
    {"red",    { {cv::Scalar(  0,100,100), cv::Scalar( 10,255,255)},
                 {cv::Scalar(160,100,100), cv::Scalar(180,255,255)} }},
    {"green",  { {cv::Scalar( 35, 60, 60), cv::Scalar( 85,255,255)} }},
    {"blue",   { {cv::Scalar(100, 80, 80), cv::Scalar(130,255,255)} }},
    {"yellow", { {cv::Scalar( 18,100,100), cv::Scalar( 35,255,255)} }},
    {"orange", { {cv::Scalar( 10,130,130), cv::Scalar( 20,255,255)} }},
    {"purple", { {cv::Scalar(125, 50, 50), cv::Scalar(155,255,255)} }},
    {"white",  { {cv::Scalar(  0,  0,180), cv::Scalar(180, 40,255)} }},
    {"black",  { {cv::Scalar(  0,  0,  0), cv::Scalar(180,255, 50)} }},
    {"brown",  { {cv::Scalar(  5, 80, 40), cv::Scalar( 20,200,200)} }},
  };
  // clang-format on

  std::string lc = color;
  std::transform(lc.begin(), lc.end(), lc.begin(), ::tolower);

  for (const auto& cd : TABLE)
  {
    if (cd.name != lc) continue;

    mask = cv::Mat::zeros(hsv.size(), CV_8U);
    for (const auto& r : cd.ranges)
    {
      cv::Mat tmp;
      cv::inRange(hsv, r.lo, r.hi, tmp);
      cv::bitwise_or(mask, tmp, mask);
    }
    return true;
  }
  return false;  // unknown color
}

// ─────────────────────────────────────────────────────────────────────────────
// Homography loader
// ─────────────────────────────────────────────────────────────────────────────
bool loadHomography(const std::string& path, cv::Mat& H)
{
  std::ifstream f(path);
  if (!f.is_open()) return false;
  H = cv::Mat(3, 3, CV_64F);
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      f >> H.at<double>(r, c);
  return f.good() || f.eof();
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera subscriber  (uses /video2 by default)
// ─────────────────────────────────────────────────────────────────────────────
class FrameSubscriber : public rclcpp::Node
{
public:
  explicit FrameSubscriber(const std::string& topic)
    : Node("va_frame_subscriber")
  {
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      topic, 10,
      [this](sensor_msgs::msg::Image::SharedPtr msg)
      {
        try {
          auto ptr = cv_bridge::toCvCopy(msg, "bgr8");
          cv::Mat resized;
          cv::resize(ptr->image, resized, cv::Size(640, 480));
          std::lock_guard<std::mutex> lk(mu_);
          frame_ = resized;
        } catch (...) {}
      });
  }

  bool getFrame(cv::Mat& out)
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (frame_.empty()) return false;
    out = frame_.clone();
    return true;
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  cv::Mat   frame_;
  std::mutex mu_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Detection result
// ─────────────────────────────────────────────────────────────────────────────
struct DetectResult
{
  int    px = 0, py = 0;
  double radius     = 0.0;
  double axis_major = 0.0;
  double axis_minor = 0.0;
  double angle_deg  = 0.0;
  enum class Method { HOUGH, HULL, BBOX, CUBOID_RECT } method = Method::BBOX;
};

std::string normalizedToken(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(),
      [](unsigned char ch){ return static_cast<char>(std::tolower(ch)); });
  return value;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cylinder top-face detection  (Hough → Hull ellipse → BBOX fallback)
// ─────────────────────────────────────────────────────────────────────────────
bool detectCylinder(const cv::Mat& frame, const cv::Mat& mask,
                    DetectResult& result)
{
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return false;

  std::sort(contours.begin(), contours.end(),
      [](const auto& a, const auto& b){
        return cv::contourArea(a) > cv::contourArea(b); });

  for (const auto& contour : contours)
  {
    double area = cv::contourArea(contour);
    if (area < MIN_CONTOUR_AREA) break;

    double peri = cv::arcLength(contour, true);
    double circularity = (peri > 0.0) ? (4.0 * CV_PI * area / (peri * peri)) : 0.0;

    cv::RotatedRect contour_rr = cv::minAreaRect(contour);
    double rr_area = contour_rr.size.width * contour_rr.size.height;
    double extent = (rr_area > 0.0) ? area / rr_area : 0.0;

    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, 0.04 * peri, true);
    bool rectangular_blob =
        approx.size() == 4 &&
        cv::isContourConvex(approx) &&
        extent > RECTANGULAR_EXTENT_MIN &&
        circularity < RECTANGULAR_CIRC_MAX;

    if (rectangular_blob) continue;

    cv::Rect bbox = cv::boundingRect(contour);

    const int pad = 10;
    cv::Rect roi(
        std::max(0, bbox.x - pad),
        std::max(0, bbox.y - pad),
        std::min(frame.cols - std::max(0, bbox.x - pad), bbox.width  + 2*pad),
        std::min(frame.rows - std::max(0, bbox.y - pad), bbox.height + 2*pad));

    cv::Mat roi_mask = mask(roi);
    cv::Mat roi_gray;
    cv::cvtColor(frame(roi), roi_gray, cv::COLOR_BGR2GRAY);
    roi_gray.setTo(0, ~roi_mask);
    cv::GaussianBlur(roi_gray, roi_gray, {9,9}, 2.0);

    double exp_r = bbox.width * HOUGH_RADIUS_FRACTION;
    int    r_min = std::max(5, (int)(exp_r * (1.0 - HOUGH_RADIUS_TOLERANCE)));
    int    r_max = (int)(exp_r * (1.0 + HOUGH_RADIUS_TOLERANCE));

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(roi_gray, circles,
                     cv::HOUGH_GRADIENT_ALT,
                     HOUGH_DP, HOUGH_MIN_DIST,
                     HOUGH_PARAM1, HOUGH_PARAM2,
                     r_min, r_max);

    double bcx = (bbox.x - roi.x) + bbox.width  / 2.0;
    double bcy = (bbox.y - roi.y) + bbox.height / 2.0;
    double gx  = bbox.width  * HOUGH_CENTRE_GATE;
    double gy  = bbox.height * HOUGH_CENTRE_GATE;

    for (const auto& c : circles)
    {
      if (std::abs(c[0] - bcx) < gx && std::abs(c[1] - bcy) < gy)
      {
        result.px         = (int)(c[0] + roi.x);
        result.py         = (int)(c[1] + roi.y);
        result.radius     = c[2];
        result.axis_major = c[2];
        result.axis_minor = c[2];
        result.angle_deg  = 0.0;
        result.method     = DetectResult::Method::HOUGH;
        return true;
      }
    }

    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    if (hull.size() >= 5)
    {
      cv::RotatedRect ell = cv::fitEllipse(hull);
      result.px         = (int)ell.center.x;
      result.py         = (int)ell.center.y;
      result.radius     = (ell.size.width + ell.size.height) / 4.0;
      result.axis_major = ell.size.width  / 2.0;
      result.axis_minor = ell.size.height / 2.0;
      result.angle_deg  = ell.angle;
      result.method     = DetectResult::Method::HULL;
      return true;
    }

    result.px         = bbox.x + bbox.width  / 2;
    result.py         = bbox.y + bbox.height / 2;
    result.radius     = bbox.width / 2.0;
    result.axis_major = bbox.width  / 2.0;
    result.axis_minor = bbox.height / 2.0;
    result.angle_deg  = 0.0;
    result.method     = DetectResult::Method::BBOX;
    return true;
  }

  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cuboid top-face detection
//   1. Scan colored contours from largest to smallest
//   2. Compute circularity  (4π·area / perimeter²)
//   3. Fit minAreaRect; check aspect ratio
//   4. If it passes both tests → it's a cuboid top face
// ─────────────────────────────────────────────────────────────────────────────
bool detectCuboid(const cv::Mat& /*frame*/, const cv::Mat& mask,
                  DetectResult& result)
{
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) return false;

  std::sort(contours.begin(), contours.end(),
      [](const auto& a, const auto& b){
        return cv::contourArea(a) > cv::contourArea(b); });

  for (const auto& contour : contours)
  {
    double area = cv::contourArea(contour);
    if (area < MIN_CONTOUR_AREA) break;

    double peri        = cv::arcLength(contour, true);
    double circularity = (peri > 0) ? (4.0 * CV_PI * area / (peri * peri)) : 0.0;

    cv::RotatedRect rr = cv::minAreaRect(contour);
    double w  = rr.size.width;
    double h  = rr.size.height;
    double ar = (h > 0) ? w / h : 1.0;
    if (ar < 1.0) ar = 1.0 / ar;   // always ≥ 1

    // Cuboid: rectangular (low circularity) and sensible aspect ratio
    if (circularity > CUBOID_CIRC_MAX) continue;   // too round
    if (ar < CUBOID_AR_MIN || ar > CUBOID_AR_MAX) continue;

    result.px         = (int)rr.center.x;
    result.py         = (int)rr.center.y;
    result.radius     = std::max(w, h) / 2.0;
    result.axis_major = std::max(w, h) / 2.0;
    result.axis_minor = std::min(w, h) / 2.0;
    result.angle_deg  = rr.angle;
    result.method     = DetectResult::Method::CUBOID_RECT;
    return true;
  }

  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified detect: routes by shape parameter
// ─────────────────────────────────────────────────────────────────────────────
bool detect(const cv::Mat& frame,
            const std::string& color, const std::string& shape,
            DetectResult& result)
{
  cv::Mat hsv, mask;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  if (!colorMask(hsv, color, mask)) return false;

  cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                   cv::Mat::ones(5, 5, CV_8U));

  if (shape == "cylinder") return detectCylinder(frame, mask, result);
  if (shape == "cuboid")   return detectCuboid  (frame, mask, result);
  return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Overlay drawing
// ─────────────────────────────────────────────────────────────────────────────
void drawOverlay(cv::Mat& frame, const DetectResult& r,
                 double rx, double ry,
                 const std::string& color, const std::string& shape)
{
  cv::Scalar col;
  switch (r.method) {
    case DetectResult::Method::HOUGH:       col = {0,220,0};   break;
    case DetectResult::Method::HULL:        col = {0,165,255}; break;
    case DetectResult::Method::CUBOID_RECT: col = {255,128,0}; break;
    default:                                col = {0,0,255};   break;
  }

  if (r.method == DetectResult::Method::HOUGH)
  {
    int cr = (int)r.radius;
    cv::circle(frame, {r.px, r.py}, cr, col, 2);
    cv::line(frame, {r.px-cr, r.py}, {r.px+cr, r.py}, col, 1);
    cv::line(frame, {r.px, r.py-cr}, {r.px, r.py+cr}, col, 1);
  }
  else if (r.method == DetectResult::Method::CUBOID_RECT)
  {
    // Draw rotated rectangle
    cv::RotatedRect rr(cv::Point2f((float)r.px, (float)r.py),
                       cv::Size2f((float)(r.axis_major*2),
                                  (float)(r.axis_minor*2)),
                       (float)r.angle_deg);
    cv::Point2f pts[4]; rr.points(pts);
    for (int i = 0; i < 4; i++)
      cv::line(frame, pts[i], pts[(i+1)%4], col, 2);
  }
  else
  {
    if (r.axis_major > 1.0)
      cv::ellipse(frame, {r.px, r.py},
                  {(int)r.axis_major, (int)r.axis_minor},
                  r.angle_deg, 0, 360, col, 2);
  }

  cv::circle(frame, {r.px, r.py}, 5, col, -1);
  cv::circle(frame, {r.px, r.py}, 16, col, 1);

  const char* tag =
      (r.method == DetectResult::Method::HOUGH)       ? "[H]" :
      (r.method == DetectResult::Method::HULL)         ? "[E]" :
      (r.method == DetectResult::Method::CUBOID_RECT)  ? "[C]" : "[B]";

  std::string label =
      std::string(tag) + " " + color + " " + shape +
      "  px:(" + std::to_string(r.px) + "," + std::to_string(r.py) + ")" +
      "  R:(" + std::to_string(rx).substr(0,6) + "," +
               std::to_string(ry).substr(0,6) + ")" +
      "  r=" + std::to_string((int)r.radius) + "px";

  cv::putText(frame, label, {r.px + 20, r.py - 10},
              cv::FONT_HERSHEY_SIMPLEX, 0.42, col, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// IO helpers (suction gripper)
// ─────────────────────────────────────────────────────────────────────────────
bool writeIO(rclcpp::Client<WriteSingleIO>::SharedPtr client,
             const rclcpp::Logger& logger,
             uint32_t address, int32_t value, int timeout_ms = 3000)
{
  auto req = std::make_shared<WriteSingleIO::Request>();
  req->address = address;
  req->value   = value;
  auto fut = client->async_send_request(req);
  if (fut.wait_for(std::chrono::milliseconds(timeout_ms))
      != std::future_status::ready)
  {
    RCLCPP_ERROR(logger, "  ❌ WriteSingleIO timeout addr=%u val=%d",
                 address, value);
    return false;
  }
  auto res = fut.get();
  if (!res->success)
  {
    RCLCPP_ERROR(logger, "  ❌ WriteSingleIO failed addr=%u val=%d msg=%s",
                 address, value, res->message.c_str());
    return false;
  }
  RCLCPP_INFO(logger, "  ✔ WriteSingleIO addr=%u val=%d OK", address, value);
  return true;
}

bool setSuction(rclcpp::Client<WriteSingleIO>::SharedPtr client,
                const rclcpp::Logger& logger, bool on)
{
  RCLCPP_INFO(logger, "  🌀 Suction %s  (IO#%u=%d)",
              on ? "ON" : "OFF", IO_SUCTION, on ? 1 : 0);
  if (!writeIO(client, logger, IO_SUCTION, on ? 1 : 0)) return false;
  rclcpp::sleep_for(std::chrono::milliseconds(600));
  RCLCPP_INFO(logger, "  ✔ Suction %s done.", on ? "ON" : "OFF");
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cartesian / incremental move helpers  (identical to green_tracker_moveit)
// ─────────────────────────────────────────────────────────────────────────────
bool cartesianMove(MoveGroupInterface& mg,
                   const geometry_msgs::msg::Pose& target,
                   const rclcpp::Logger& logger,
                   const std::string& name,
                   double threshold = 0.95,
                   double pos_tol   = 0.05)
{
  std::vector<geometry_msgs::msg::Pose> wp{target};
  moveit_msgs::msg::RobotTrajectory traj;
  double frac = mg.computeCartesianPath(wp, 0.01, 0.0, traj);

  RCLCPP_INFO(logger, "[%s] coverage=%.1f%%  wpts=%zu",
              name.c_str(), frac*100.0,
              traj.joint_trajectory.points.size());

  if (frac < threshold) {
    RCLCPP_ERROR(logger, "[%s] ❌ planning failed %.1f%%", name.c_str(), frac*100.0);
    return false;
  }

  MoveGroupInterface::Plan plan;
  plan.trajectory_ = traj;
  auto res = mg.execute(plan);
  mg.stop();

  auto after = mg.getCurrentPose().pose;
  double err = std::sqrt(
      std::pow(after.position.x - target.position.x, 2) +
      std::pow(after.position.y - target.position.y, 2) +
      std::pow(after.position.z - target.position.z, 2));

  RCLCPP_INFO(logger, "[%s] err=%.4fm", name.c_str(), err);

  if (res != moveit::core::MoveItErrorCode::SUCCESS) {
    if (err < pos_tol) {
      RCLCPP_WARN(logger, "[%s] ⚠ err<%.3f → SUCCESS (quirk)", name.c_str(), pos_tol);
      return true;
    }
    RCLCPP_ERROR(logger, "[%s] ❌ err=%.4f>%.3f → FAILED", name.c_str(), err, pos_tol);
    return false;
  }
  RCLCPP_INFO(logger, "[%s] ✔ Done.", name.c_str());
  return true;
}

bool incrementalMove(
    MoveGroupInterface& mg,
    const geometry_msgs::msg::Pose& target,
    const rclcpp::Logger& logger,
    const std::string& name,
    double step_size = 0.05,
    rclcpp::Client<motoros2_interfaces::srv::StartTrajMode>::SharedPtr
        traj_client = nullptr)
{
  auto start = mg.getCurrentPose().pose;
  double dx = target.position.x - start.position.x;
  double dy = target.position.y - start.position.y;
  double dz = target.position.z - start.position.z;
  double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
  int steps = std::max(1, (int)std::ceil(dist / step_size));

  RCLCPP_INFO(logger, "[%s] dist=%.4fm steps=%d", name.c_str(), dist, steps);

  for (int i = 1; i <= steps; i++)
  {
    double a = (double)i / steps;
    geometry_msgs::msg::Pose p = start;
    p.position.x = start.position.x + dx * a;
    p.position.y = start.position.y + dy * a;
    p.position.z = start.position.z + dz * a;

    if (traj_client) {
      traj_client->async_send_request(
          std::make_shared<motoros2_interfaces::srv::StartTrajMode::Request>());
      rclcpp::sleep_for(std::chrono::milliseconds(300));
    }

    std::string sname = name + "-" + std::to_string(i) + "/" + std::to_string(steps);
    if (!cartesianMove(mg, p, logger, sname)) {
      RCLCPP_ERROR(logger, "[%s] ❌ failed at step %d", name.c_str(), i);
      return false;
    }
  }
  RCLCPP_INFO(logger, "[%s] ✔ All %d steps complete.", name.c_str(), steps);
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>(
      "voice_assistant",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto logger = node->get_logger();

  // ── Declare parameters ─────────────────────────────────────────────────────
  auto declare_safe = [&](const std::string& n, auto v) {
    if (!node->has_parameter(n)) node->declare_parameter(n, v);
  };

  declare_safe("color",           std::string("green"));
  declare_safe("shape",           std::string("cylinder"));
  declare_safe("homography_path", std::string("/home/blessinggrace/voice_assistant/homography.txt"));
  declare_safe("camera_topic",    std::string("/video_frames"));   // mapped to /video2
  declare_safe("step_size",       0.05);
  declare_safe("target_z",        DEFAULT_PICK_Z);
  declare_safe("drop_z",          DEFAULT_DROP_Z);
  declare_safe("min_z",           DEFAULT_MIN_Z);

  std::string color       = node->get_parameter("color").as_string();
  std::string shape       = node->get_parameter("shape").as_string();
  std::string hpath       = node->get_parameter("homography_path").as_string();
  std::string cam_topic   = node->get_parameter("camera_topic").as_string();
  double      step_size   = node->get_parameter("step_size").as_double();
  double      target_z    = node->get_parameter("target_z").as_double();
  double      drop_z      = node->get_parameter("drop_z").as_double();
  double      min_z       = node->get_parameter("min_z").as_double();

  color = normalizedToken(color);
  shape = normalizedToken(shape);

  if (target_z < min_z) {
    RCLCPP_WARN(logger,
        "target_z %.4fm is below min_z %.4fm; clamping target_z to min_z.",
        target_z, min_z);
    target_z = min_z;
  }
  if (drop_z < min_z) {
    RCLCPP_WARN(logger,
        "drop_z %.4fm is below min_z %.4fm; clamping drop_z to min_z.",
        drop_z, min_z);
    drop_z = min_z;
  }

  RCLCPP_INFO(logger, "════════════════════════════════════════");
  RCLCPP_INFO(logger, "  voice_assistant");
  RCLCPP_INFO(logger, "  color      : %s", color.c_str());
  RCLCPP_INFO(logger, "  shape      : %s", shape.c_str());
  RCLCPP_INFO(logger, "  camera     : %s", cam_topic.c_str());
  RCLCPP_INFO(logger, "  homography : %s", hpath.c_str());
  RCLCPP_INFO(logger, "  step_size  : %.3fm", step_size);
  RCLCPP_INFO(logger, "  target_z   : %.4fm", target_z);
  RCLCPP_INFO(logger, "  drop_z     : %.4fm", drop_z);
  RCLCPP_INFO(logger, "  min_z      : %.4fm", min_z);
  RCLCPP_INFO(logger, "════════════════════════════════════════");

  // ── Validate color / shape ─────────────────────────────────────────────────
  {
    cv::Mat dummy_hsv(1,1,CV_8UC3, cv::Scalar(0,0,0));
    cv::Mat dummy_mask;
    if (!colorMask(dummy_hsv, color, dummy_mask)) {
      RCLCPP_ERROR(logger, "❌ Unknown color: '%s'", color.c_str());
      return 1;
    }
  }
  if (shape != "cylinder" && shape != "cuboid") {
    RCLCPP_ERROR(logger, "❌ Unknown shape: '%s'  (use cylinder or cuboid)", shape.c_str());
    return 1;
  }

  // ── Homography ─────────────────────────────────────────────────────────────
  cv::Mat H;
  if (!loadHomography(hpath, H)) {
    RCLCPP_ERROR(logger, "❌ Failed to load homography: %s", hpath.c_str());
    return 1;
  }
  RCLCPP_INFO(logger, "✔ Homography loaded.");

  // ── Executor ───────────────────────────────────────────────────────────────
  auto frame_node = std::make_shared<FrameSubscriber>(cam_topic);

  rclcpp::executors::MultiThreadedExecutor exec;
  exec.add_node(node);
  exec.add_node(frame_node);
  std::thread spin_thread([&]{ exec.spin(); });

  // ── Wait for camera ────────────────────────────────────────────────────────
  RCLCPP_INFO(logger, "Waiting for camera on %s …", cam_topic.c_str());
  {
    cv::Mat tmp;
    int waited = 0;
    while (rclcpp::ok() && !frame_node->getFrame(tmp)) {
      rclcpp::sleep_for(std::chrono::milliseconds(200));
      if (++waited % 10 == 0)
        RCLCPP_WARN(logger, "Still waiting for camera (%ds) …", waited/5);
      if (waited > 100) {
        RCLCPP_ERROR(logger, "❌ No camera after 20s on %s", cam_topic.c_str());
        exec.cancel(); spin_thread.join();
        return 1;
      }
    }
  }
  RCLCPP_INFO(logger, "✔ Camera ready.");

  // ── Trajectory mode ────────────────────────────────────────────────────────
  auto traj_client =
      node->create_client<motoros2_interfaces::srv::StartTrajMode>(
          "/yaskawa_hc10/start_traj_mode");
  traj_client->wait_for_service();
  traj_client->async_send_request(
      std::make_shared<motoros2_interfaces::srv::StartTrajMode::Request>());
  RCLCPP_INFO(logger, "✔ Traj mode sent.");

  // ── Suction ────────────────────────────────────────────────────────────────
  auto suction_client =
      node->create_client<WriteSingleIO>("/yaskawa_hc10/write_single_io");
  suction_client->wait_for_service();
  RCLCPP_INFO(logger, "✔ Suction IO ready.");

  rclcpp::sleep_for(std::chrono::seconds(3));

  // ── MoveIt ─────────────────────────────────────────────────────────────────
  MoveGroupInterface move_group(node, "manipulator");
  move_group.setMaxVelocityScalingFactor(0.0375);
  move_group.setMaxAccelerationScalingFactor(0.0375);
  move_group.startStateMonitor();

  rclcpp::sleep_for(std::chrono::seconds(3));

  auto init_pose = move_group.getCurrentPose().pose;
  RCLCPP_INFO(logger, "✔ Robot at X=%.4f Y=%.4f Z=%.4f",
              init_pose.position.x,
              init_pose.position.y,
              init_pose.position.z);

  RCLCPP_INFO(logger, "Searching for %s %s … SPACE=lock  ESC=quit",
              color.c_str(), shape.c_str());

  // ── State machine ──────────────────────────────────────────────────────────
  enum class State { LIVE, FROZEN, DONE };
  State state = State::LIVE;

  cv::Mat       frozen_frame;
  double        target_rx = 0, target_ry = 0;
  DetectResult  frozen_result;
  int           frame_count = 0;
  geometry_msgs::msg::Pose home_pose;

  std::string win_title = "Voice Assistant: " + color + " " + shape;

  while (rclcpp::ok())
  {
    int key = cv::waitKey(30);
    if (key == 27) { RCLCPP_INFO(logger, "ESC. Quitting."); break; }

    // ── LIVE ────────────────────────────────────────────────────────────────
    if (state == State::LIVE)
    {
      cv::Mat frame;
      if (!frame_node->getFrame(frame)) continue;
      frame_count++;

      DetectResult r;
      bool found = detect(frame, color, shape, r);

      double rx = 0, ry = 0;
      if (found) {
        std::vector<cv::Point2f> src{{(float)r.px, (float)r.py}};
        std::vector<cv::Point2f> dst;
        cv::perspectiveTransform(src, dst, H);
        rx = dst[0].x;
        ry = dst[0].y;
        drawOverlay(frame, r, rx, ry, color, shape);
      }

      // crosshair
      cv::line(frame, {320-20,240}, {320+20,240}, {255,255,255}, 1);
      cv::line(frame, {320,240-20}, {320,240+20}, {255,255,255}, 1);

      cv::putText(frame,
          found ? "DETECTED - SPACE to lock" : "Searching ...",
          {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
          found ? cv::Scalar(0,220,0) : cv::Scalar(0,100,255), 1);

      cv::putText(frame,
          "Target: " + color + " " + shape + "   ESC=quit",
          {10,45}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {255,255,0}, 1);

      cv::imshow(win_title, frame);

      if (key == 32 && found) {
        home_pose     = move_group.getCurrentPose().pose;
        frozen_frame  = frame.clone();
        target_rx     = rx;
        target_ry     = ry;
        frozen_result = r;
        state         = State::FROZEN;

        RCLCPP_INFO(logger, "🔒 LOCKED  px=(%d,%d)  XY=(%.4f,%.4f)",
                    r.px, r.py, rx, ry);
        RCLCPP_INFO(logger, "   Press SPACE again to begin pick-place.");

        cv::Mat disp = frozen_frame.clone();
        cv::putText(disp, "LOCKED - SPACE to start pick-place",
            {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,165,255}, 2);
        cv::imshow(win_title, disp);

      } else if (key == 32 && !found) {
        RCLCPP_WARN(logger, "⚠ SPACE pressed but %s %s not detected.",
                    color.c_str(), shape.c_str());
      }
    }

    // ── FROZEN ──────────────────────────────────────────────────────────────
    else if (state == State::FROZEN)
    {
      cv::Mat disp = frozen_frame.clone();
      cv::putText(disp, "LOCKED - SPACE to start  ESC to quit",
          {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0,165,255}, 2);
      cv::putText(disp,
          color + " " + shape +
          "  XY:(" + std::to_string(target_rx).substr(0,6) +
          "," + std::to_string(target_ry).substr(0,6) + ")" +
          "  pickZ:" + std::to_string(target_z).substr(0,6) +
          " dropZ:" + std::to_string(drop_z).substr(0,6) + "m",
          {10,50}, cv::FONT_HERSHEY_SIMPLEX, 0.42, {0,165,255}, 1);
      cv::imshow(win_title, disp);

      if (key != 32) continue;

      // ── status overlay helper ──────────────────────────────────────────
      auto show_status = [&](const std::string& msg, cv::Scalar col) {
        cv::Mat d = frozen_frame.clone();
        cv::putText(d, msg, {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2);
        auto p = move_group.getCurrentPose().pose;
        cv::putText(d,
            "Z=" + std::to_string(p.position.z).substr(0,6) + "m",
            {10,50}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {200,200,255}, 1);
        cv::imshow(win_title, d);
        cv::waitKey(1);
      };

      auto poseAt = [&](double x, double y, double z) {
        auto p = move_group.getCurrentPose().pose;
        p.position.x = x; p.position.y = y; p.position.z = z;
        return p;
      };

      auto limitZ = [&](double requested_z, const std::string& step_name) {
        if (requested_z >= min_z) return requested_z;
        RCLCPP_WARN(logger,
            "[%s] requested Z %.4fm is below min_z %.4fm; using min_z.",
            step_name.c_str(), requested_z, min_z);
        return min_z;
      };

      RCLCPP_INFO(logger,
                  "🚀 PICK-PLACE START  %s %s  XY=(%.4f,%.4f)  pick_z=%.4f drop_z=%.4f",
                  color.c_str(), shape.c_str(), target_rx, target_ry,
                  target_z, drop_z);

      bool ok = true;

      // ── STEP 0  Suction OFF ───────────────────────────────────────────
      show_status("STEP 0: Suction OFF...", {0,165,255});
      if (!setSuction(suction_client, logger, false))
        { ok = false; show_status("STEP 0 FAILED.", {0,0,255}); goto seq_end; }

      // ── STEP 1  XY to object ──────────────────────────────────────────
      {
        show_status("STEP 1: XY to object...", {0,165,255});
        auto cur = move_group.getCurrentPose().pose;
        if (!incrementalMove(move_group,
                poseAt(target_rx, target_ry, cur.position.z),
                logger, "XY", step_size, traj_client))
          { ok = false; show_status("STEP 1 FAILED.", {0,0,255}); goto seq_end; }
      }

      // ── STEP 2  Z descend ─────────────────────────────────────────────
      {
        show_status("STEP 2: Descending...", {0,165,255});
        auto cur = move_group.getCurrentPose().pose;
        if (!incrementalMove(move_group,
                poseAt(cur.position.x, cur.position.y,
                       limitZ(target_z, "Z_down")),
                logger, "Z_down", step_size, traj_client))
          { ok = false; show_status("STEP 2 FAILED.", {0,0,255}); goto seq_end; }
      }

      // ── STEP 3  Suction ON ────────────────────────────────────────────
      show_status("STEP 3: Suction ON...", {0,165,255});
      if (!setSuction(suction_client, logger, true))
        { ok = false; show_status("STEP 3 FAILED.", {0,0,255}); goto seq_end; }

      // ── STEP 4  Lift ──────────────────────────────────────────────────
      {
        show_status("STEP 4: Lifting...", {0,165,255});
        auto cur = move_group.getCurrentPose().pose;
        if (!incrementalMove(move_group,
                poseAt(cur.position.x, cur.position.y,
                       cur.position.z + LIFT_HEIGHT),
                logger, "Z_lift", step_size, traj_client))
          { ok = false; show_status("STEP 4 FAILED.", {0,0,255}); goto seq_end; }
      }

      // ── STEP 5  XY to place ───────────────────────────────────────────
      {
        show_status("STEP 5: XY to place...", {0,165,255});
        auto cur = move_group.getCurrentPose().pose;
        if (!incrementalMove(move_group,
                poseAt(target_rx + PLACE_OFFSET_X,
                       target_ry + PLACE_OFFSET_Y,
                       cur.position.z),
                logger, "XY_place", step_size, traj_client))
          { ok = false; show_status("STEP 5 FAILED.", {0,0,255}); goto seq_end; }
      }

      // ── STEP 5b  Lower to drop height ─────────────────────────────────
      {
        show_status("STEP 5b: Lowering to drop height...", {0,165,255});
        auto cur = move_group.getCurrentPose().pose;
        if (!incrementalMove(move_group,
                poseAt(cur.position.x, cur.position.y,
                       limitZ(drop_z, "Z_lower")),
                logger, "Z_lower", step_size, traj_client))
          { ok = false; show_status("STEP 5b FAILED.", {0,0,255}); goto seq_end; }
      }

      // ── STEP 6  Suction OFF ───────────────────────────────────────────
      show_status("STEP 6: Suction OFF (release)...", {0,165,255});
      if (!setSuction(suction_client, logger, false))
        { ok = false; show_status("STEP 6 FAILED.", {0,0,255}); goto seq_end; }

      // ── STEP 7  Retreat ───────────────────────────────────────────────
      {
        show_status("STEP 7: Retreating...", {0,165,255});
        auto cur = move_group.getCurrentPose().pose;
        if (!incrementalMove(move_group,
                poseAt(cur.position.x, cur.position.y,
                       cur.position.z + RETREAT_HEIGHT),
                logger, "Z_retreat", step_size, traj_client))
          { ok = false; show_status("STEP 7 FAILED.", {0,0,255}); goto seq_end; }
      }

      // ── STEP 8  Return home ───────────────────────────────────────────
      {
        show_status("STEP 8: Returning home...", {0,165,255});
        auto cur = move_group.getCurrentPose().pose;

        // 8a: raise Z first
        if (!incrementalMove(move_group,
                poseAt(cur.position.x, cur.position.y, home_pose.position.z),
                logger, "Home_Z", step_size, traj_client))
          { ok = false; show_status("STEP 8a FAILED.", {0,0,255}); goto seq_end; }

        // 8b: XY home
        if (!incrementalMove(move_group,
                poseAt(home_pose.position.x, home_pose.position.y,
                       home_pose.position.z),
                logger, "Home_XY", step_size, traj_client))
          { ok = false; show_status("STEP 8b FAILED.", {0,0,255}); goto seq_end; }
      }

      seq_end:
      {
        auto fp = move_group.getCurrentPose().pose;
        if (ok)
          RCLCPP_INFO(logger, "✔ PICK-PLACE COMPLETE.  Final Z=%.4f", fp.position.z);
        else
          RCLCPP_ERROR(logger, "❌ SEQUENCE FAILED.  Final Z=%.4f", fp.position.z);

        state = State::DONE;
        show_status(
            ok ? "DONE. Pick-place complete. ESC to quit."
               : "SEQUENCE FAILED. ESC to quit.",
            ok ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255));
      }
    }

    // ── DONE ────────────────────────────────────────────────────────────────
    else if (state == State::DONE)
    {
      cv::Mat disp = frozen_frame.clone();
      cv::putText(disp, "DONE. ESC to quit.",
          {10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,255,0}, 2);
      cv::imshow(win_title, disp);
    }
  }

  cv::destroyAllWindows();
  exec.cancel();
  spin_thread.join();
  rclcpp::shutdown();
  return 0;
}
