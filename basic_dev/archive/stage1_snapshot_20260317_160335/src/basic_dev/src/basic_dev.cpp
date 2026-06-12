#ifndef _BASIC_DEV_CPP_
#define _BASIC_DEV_CPP_

#include "basic_dev.hpp"
#include <fstream>
#include <sstream>
#include <ros/package.h>
#include <queue>

namespace {

// File-local helpers for stage2 planning. These are kept outside the class
// because they are pure implementation detail and do not depend on ROS state.
struct Stage2CandidateCell {
    double cost = 0.0;
    int pitch_idx = 0;
    int yaw_idx = 0;
};

inline double stage2_angle_diff(double a, double b)
{
    return std::fabs(std::atan2(std::sin(a - b), std::cos(a - b)));
}

double point_to_segment_distance(const Eigen::Vector3d& p,
                                 const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b)
{
    const Eigen::Vector3d ab = b - a;
    const double ab2 = ab.squaredNorm();
    if (ab2 <= 1e-9) return (p - a).norm();
    const double t = std::max(0.0, std::min(1.0, (p - a).dot(ab) / ab2));
    return (p - (a + t * ab)).norm();
}

void smooth_stage2_polar_matrix(Eigen::MatrixXf& matrix, int pitch_radius, int yaw_radius)
{
    if (matrix.rows() <= 0 || matrix.cols() <= 0) return;
    Eigen::MatrixXf input = matrix;
    for (int p = 0; p < matrix.rows(); ++p) {
        for (int y = 0; y < matrix.cols(); ++y) {
            double weighted_sum = 0.0;
            double weight_total = 0.0;
            for (int dp = -pitch_radius; dp <= pitch_radius; ++dp) {
                const int pp = p + dp;
                if (pp < 0 || pp >= matrix.rows()) continue;
                for (int dy = -yaw_radius; dy <= yaw_radius; ++dy) {
                    int yy = y + dy;
                    while (yy < 0) yy += matrix.cols();
                    while (yy >= matrix.cols()) yy -= matrix.cols();
                    const double weight =
                        static_cast<double>(pitch_radius - std::abs(dp) + 1) *
                        static_cast<double>(yaw_radius - std::abs(dy) + 1);
                    weighted_sum += weight * static_cast<double>(input(pp, yy));
                    weight_total += weight;
                }
            }
            matrix(p, y) = (weight_total > 0.0)
                               ? static_cast<float>(weighted_sum / weight_total)
                               : input(p, y);
        }
    }
}

}  // namespace

int main(int argc, char** argv)
{

    ros::init(argc, argv, "basic_dev"); // 初始化ros 节点，命名为 basic
    ros::NodeHandle n; // 创建node控制句柄
    BasicDev go(&n);
    return 0;
}

// -----------------------------------------------------------------------------
// Constructor / ROS wiring / parameter loading
// -----------------------------------------------------------------------------
BasicDev::BasicDev(ros::NodeHandle *nh)
{  
    nh_ = nh;
    // ------------------------------
    // 初始化图像传输与缓存
    // ------------------------------
    it = std::make_unique<image_transport::ImageTransport>(*nh); 
    front_left_img = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0));
    front_right_img = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0));
    back_left_img = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0));
    back_right_img = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0));

    // ------------------------------
    // 服务请求默认参数
    // ------------------------------
    takeoff.request.waitOnLastTask = 1;
    land.request.waitOnLastTask = 1;

    // ------------------------------
    // 7字段速度控制消息默认值
    // ------------------------------
    velcmd.vx = 0;
    velcmd.vy = 0;
    velcmd.vz = 0;
    velcmd.yawRate = 0;
    velcmd.va = 10;
    velcmd.stop = 0;
    velcmd.header.stamp = ros::Time::now();

    // ------------------------------
    // 状态输入订阅
    // ------------------------------
    odom_suber = nh->subscribe<geometry_msgs::PoseStamped>("/airsim_node/drone_1/debug/pose_gt", 1, std::bind(&BasicDev::pose_cb, this, std::placeholders::_1));
    gps_suber = nh->subscribe<geometry_msgs::PoseStamped>("/airsim_node/drone_1/gps", 1, std::bind(&BasicDev::gps_cb, this, std::placeholders::_1));
    imu_suber = nh->subscribe<sensor_msgs::Imu>("/airsim_node/drone_1/imu/imu", 1, std::bind(&BasicDev::imu_cb, this, std::placeholders::_1));
    lidar_suber = nh->subscribe<sensor_msgs::PointCloud2>("/airsim_node/drone_1/lidar", 1, std::bind(&BasicDev::lidar_cb, this, std::placeholders::_1));
    pwm_feedback_suber = nh->subscribe<airsim_ros::RotorPWM>("/airsim_node/drone_1/debug/rotor_pwm", 1, std::bind(&BasicDev::pwm_feedback_cb, this, std::placeholders::_1));
    initial_pose_suber = nh->subscribe<geometry_msgs::PoseStamped>("/airsim_node/initial_pose", 1, std::bind(&BasicDev::initial_pose_cb, this, std::placeholders::_1));
    end_goal_suber = nh->subscribe<geometry_msgs::PoseStamped>("/airsim_node/end_goal", 1, std::bind(&BasicDev::end_goal_cb, this, std::placeholders::_1));

    // ------------------------------
    // 相机订阅（调试用）
    // ------------------------------
    front_left_view_suber = it->subscribe("/airsim_node/drone_1/front_left/Scene", 1, std::bind(&BasicDev::front_left_view_cb, this,  std::placeholders::_1));
    front_right_view_suber = it->subscribe("/airsim_node/drone_1/front_right/Scene", 1, std::bind(&BasicDev::front_right_view_cb, this,  std::placeholders::_1));
    back_left_view_suber = it->subscribe("/airsim_node/drone_1/back_left/Scene", 1, std::bind(&BasicDev::back_left_view_cb, this,  std::placeholders::_1));
    back_right_view_suber = it->subscribe("/airsim_node/drone_1/back_right/Scene", 1, std::bind(&BasicDev::back_right_view_cb, this,  std::placeholders::_1));

    // ------------------------------
    // 服务客户端与控制发布器
    // ------------------------------
    takeoff_client = nh->serviceClient<airsim_ros::Takeoff>("/airsim_node/drone_1/takeoff");
    land_client = nh->serviceClient<airsim_ros::Land>("/airsim_node/drone_1/land");
    reset_client = nh->serviceClient<airsim_ros::Reset>("/airsim_node/reset");
    vel_publisher = nh->advertise<airsim_ros::VelCmd>("/airsim_node/drone_1/vel_body_cmd", 1);
    desired_path_pub_ = nh->advertise<nav_msgs::Path>("/viz/desired_path", 1, true);
    actual_path_pub_ = nh->advertise<nav_msgs::Path>("/viz/actual_path", 1, false);
    target_marker_pub_ = nh->advertise<visualization_msgs::Marker>("/viz/target_point", 1, false);
    corridor_map_pub_ = nh->advertise<sensor_msgs::PointCloud2>("/viz/corridor_map", 1, false);
    lidar_frame_pub_ = nh->advertise<sensor_msgs::PointCloud2>("/viz/lidar_frame_world", 1, false);
    stage2_obstacle_points_pub_ = nh->advertise<sensor_msgs::PointCloud2>("/viz/stage2_obstacle_points", 1, false);
    stage2_debug_marker_pub_ = nh->advertise<visualization_msgs::Marker>("/viz/stage2_avoid_debug", 8, false);

    // 33Hz 控制定时器
    control_timer = nh->createTimer(ros::Duration(0.03), &BasicDev::control_timer_cb, this);

    // 竞速默认参数（Z轴按传统单环PID）
    // 可通过参数调整测试速度和动作幅度
    nh->param("test_speed_xy", test_speed_xy, 12.0);         // 测试平移阶段的平面速度（m/s）
    nh->param("test_yaw_rate_deg", test_yaw_rate_deg, 45.0); // 测试旋转阶段角速度（deg/s）
    nh->param("test_move_dist", test_move_dist, 15.0);       // 测试平移阶段目标位移（m）
    nh->param("test_hover_sec", test_hover_sec, 0.0);        // 起飞后首次悬停时间（s）
    nh->param("buffer_dist", buffer_dist_, 10.0);            // 路径跟踪前的前飞缓冲距离（m）
    nh->param("buffer_vx", buffer_vx_, 16.0);                // 前飞缓冲段机体系前向速度（m/s）
    nh->param("buffer_vz", buffer_vz_, 0.8);                 // 前飞缓冲段机体系竖直速度（m/s）
    nh->param("buffer_max_climb", buffer_max_climb_, 0.9);   // 前飞缓冲段允许的最大相对爬升（m）
    nh->param("buffer_hover_sec", buffer_hover_sec_, 0.0);   // 前飞缓冲结束后的悬停时间（s）
    nh->param("max_vx", max_vx_, 18.0);                      // 路径跟踪时机体系X轴速度上限（m/s）
    nh->param("max_vy", max_vy_, 5.0);                      // 路径跟踪时机体系Y轴速度上限（m/s）
    nh->param("max_vz", max_vz_, 12.0);                 // 路径跟踪时机体系Z轴上升速度上限（m/s）
    nh->param("max_vz_start_zone", max_vz_start_zone_, 2.6); // 起点后前段Z速度上限（m/s）

    // CSV轨迹跟踪参数
    // 轨迹CSV默认路径：优先使用宿主机绝对路径，其次包内文件，再次镜像根目录
    std::string default_csv_path = "/home/peter/RMUA2026-LT/IntelligentUAVChampionshipBase-RMUA2026/basic_dev/gnss_path_fast_aligned.csv";
    std::string default_stage2_csv_path = "/home/peter/RMUA2026-LT/IntelligentUAVChampionshipBase-RMUA2026/basic_dev/赛道3反向_smoothed_yflip.csv";
    const std::string pkg_path = ros::package::getPath("basic_dev");
    if (!pkg_path.empty()) {
        const std::string stage1_candidate = pkg_path + "/gnss_path_fast_aligned.csv";
        if (std::ifstream(stage1_candidate).good()) {
            default_csv_path = stage1_candidate;
        }
        const std::string stage2_candidate = pkg_path + "/赛道3反向_smoothed_yflip.csv";
        if (std::ifstream(stage2_candidate).good()) {
            default_stage2_csv_path = stage2_candidate;
        }
    }
    if (!std::ifstream(default_csv_path).good()) {
        const std::string stage1_container_candidate = "/basic_dev/gnss_path_fast_aligned.csv";
        if (std::ifstream(stage1_container_candidate).good()) {
            default_csv_path = stage1_container_candidate;
        }
    }
    if (!std::ifstream(default_stage2_csv_path).good()) {
        const std::string stage2_container_candidate = "/basic_dev/赛道3反向_smoothed_yflip.csv";
        if (std::ifstream(stage2_container_candidate).good()) {
            default_stage2_csv_path = stage2_container_candidate;
        }
    }
    nh->param("csv_path", csv_path_, default_csv_path);
    nh->param("stage2_csv_path", stage2_csv_path_, default_stage2_csv_path);
    const bool csv_exists = std::ifstream(csv_path_).good();
    const bool stage2_csv_exists = std::ifstream(stage2_csv_path_).good();
    ROS_INFO("csv_path=%s (exists=%d)", csv_path_.c_str(), static_cast<int>(csv_exists));
    ROS_INFO("stage2_csv_path=%s (exists=%d)", stage2_csv_path_.c_str(), static_cast<int>(stage2_csv_exists));
    if (!csv_exists) {
        ROS_WARN("csv_path file not found at startup; will try loading anyway when entering FOLLOW_PATH.");
    }
    if (!stage2_csv_exists) {
        ROS_WARN("stage2_csv_path file not found at startup; stage2 handoff will fail until the file exists.");
    }
    nh->param("downsample_dist", downsample_dist_, 0.4);   // 轨迹点降采样间距（m）
    nh->param("lookahead_dist", lookahead_dist_, 6.5);    // 前视距离基线（m）
    nh->param("lookahead_min_dist", lookahead_min_dist_, 6.0); // 自适应前视最小距离（m）
    nh->param("lookahead_max_dist", lookahead_max_dist_, 10.0); // 自适应前视最大距离（m）
    nh->param("lookahead_time_gain", lookahead_time_gain_, 0.25); // 自适应前视速度增益（m / (m/s)）
    nh->param("lookahead_speed_ref", lookahead_speed_ref_, 10.0); // 超过该水平速度后再逐步放大前视（m/s）
    nh->param("reach_end_dist", reach_end_dist_, 0.6);     // 终点判定阈值（m）
    nh->param("final_brake_start_dist", final_brake_start_dist_, 16.0); // 终点前开始提前刹车的距离（m）
    nh->param("final_brake_min_scale", final_brake_min_scale_, 0.20); // 提前刹车时保留的最小速度比例
    nh->param("final_stop_before_end_dist", final_stop_before_end_dist_, 8.0); // 终点前提前停下的距离（m）
    nh->param("final_turn_deg", final_turn_deg_, 180.0); // 终点前悬停后原地转向角度（deg）
    nh->param("final_turn_rate_limit_deg", final_turn_rate_limit_deg_, 35.0); // 终点原地转向最大角速度（deg/s）
    nh->param("final_turn_tolerance_deg", final_turn_tolerance_deg_, 2.0); // 终点原地转向完成判定（deg）
    nh->param("enable_stage2", enable_stage2_, true);     // 掉头完成后是否切换到第二赛段
    nh->param("stage2_manual_takeover_enable", stage2_manual_takeover_enable_, false); // 掉头后是否停在人工接管等待阶段
    nh->param("stage2_invert_csv_y", stage2_invert_csv_y_, false); // 第二赛段CSV是否额外翻转Y
    nh->param("stage2_avoid_enable", stage2_avoid_enable_, true); // 第二赛段是否启用第一步前向避障
    nh->param("stage2_max_vx", stage2_max_vx_, 8.0); // 第二赛段X轴慢速档上限（m/s）
    nh->param("stage2_max_vy", stage2_max_vy_, 4.0); // 第二赛段常规横向速度上限（m/s）
    nh->param("stage2_max_vz", stage2_max_vz_, 8.0); // 第二赛段常规竖向速度上限（m/s）
    nh->param("stage2_avoid_max_vy", stage2_avoid_max_vy_, 12.0); // 第二赛段避障时横向速度上限（m/s）
    nh->param("stage2_avoid_max_vz", stage2_avoid_max_vz_, 10.0); // 第二赛段避障时竖向速度上限（m/s）
    nh->param("stage2_avoid_vy_gain_scale", stage2_avoid_vy_gain_scale_, 1.8); // 第二赛段避障时Y控制增益倍率
    nh->param("stage2_avoid_vz_gain_scale", stage2_avoid_vz_gain_scale_, 1.4); // 第二赛段避障时Z控制增益倍率
    nh->param("stage2_avoid_min_vy_cmd", stage2_avoid_min_vy_cmd_, 3.5); // 第二赛段避障时最小横向执行速度（m/s）
    nh->param("stage2_recover_max_vz", stage2_recover_max_vz_, 2.0); // 第二赛段恢复高度时单独限幅（m/s）
    nh->param("stage2_avoid_vy_slew_rate", stage2_avoid_vy_slew_rate_, 12.0); // 第二赛段横向指令变化率限幅（m/s^2）
    nh->param("stage2_avoid_vz_slew_rate", stage2_avoid_vz_slew_rate_, 10.0); // 第二赛段竖向指令变化率限幅（m/s^2）
    nh->param("stage2_avoid_max_pitch_deg", stage2_avoid_max_pitch_deg_, 18.0); // 第二赛段局部规划允许的最大俯仰绕障角（deg）
    nh->param("stage2_avoid_vertical_weight", stage2_avoid_vertical_weight_, 1.1); // 第二赛段局部规划：竖向偏移代价权重
    nh->param("stage2_avoid_vertical_rejoin_weight", stage2_avoid_vertical_rejoin_weight_, 0.8); // 第二赛段局部规划：回接全局高度代价权重
    nh->param("stage2_avoid_candidate_switch_margin", stage2_avoid_candidate_switch_margin_, 3.0); // 第二赛段候选切换迟滞，越大越不容易换层/换方向
    nh->param("stage2_avoid_target_xy_slew_rate", stage2_avoid_target_xy_slew_rate_, 6.0); // 第二赛段局部目标XY变化率限幅（m/s）
    nh->param("stage2_avoid_target_z_slew_rate", stage2_avoid_target_z_slew_rate_, 2.5); // 第二赛段局部目标Z变化率限幅（m/s）
    nh->param("stage2_z_avoid_trigger_frames", stage2_z_avoid_trigger_frames_, 3); // 第二赛段Z绕障接管前要求连续触发的帧数
    nh->param("stage2_dynamic_avoid_max_vy", stage2_dynamic_avoid_max_vy_, 12.0); // 第二赛段迎面动态障碍时横向速度上限（m/s）
    nh->param("stage2_dynamic_avoid_max_vz", stage2_dynamic_avoid_max_vz_, 12.0); // 第二赛段迎面动态障碍时竖向速度上限（m/s）
    nh->param("stage2_dynamic_avoid_vy_gain_scale", stage2_dynamic_avoid_vy_gain_scale_, 2.4); // 第二赛段迎面动态障碍时Y控制增益倍率
    nh->param("stage2_dynamic_avoid_vz_gain_scale", stage2_dynamic_avoid_vz_gain_scale_, 1.9); // 第二赛段迎面动态障碍时Z控制增益倍率
    nh->param("stage2_dynamic_avoid_vy_slew_rate", stage2_dynamic_avoid_vy_slew_rate_, 18.0); // 第二赛段迎面动态障碍时横向变化率限幅（m/s^2）
    nh->param("stage2_dynamic_avoid_vz_slew_rate", stage2_dynamic_avoid_vz_slew_rate_, 16.0); // 第二赛段迎面动态障碍时竖向变化率限幅（m/s^2）
    nh->param("stage2_dynamic_avoid_max_lateral_offset", stage2_dynamic_avoid_max_lateral_offset_, 1.2); // 第二赛段迎面动态障碍时局部目标最大横向偏移（m）
    nh->param("stage2_z_blend_dist", stage2_z_blend_dist_, 4.0); // 第二赛段正式跟踪初段的高度过渡距离（m）
    nh->param("stage2_avoid_forward_range", stage2_avoid_forward_range_, 12.0); // 第二赛段前向检测距离（m）
    nh->param("stage2_avoid_lateral_half_width", stage2_avoid_lateral_half_width_, 1.8); // 第二赛段前向检测横向半宽（m）
    nh->param("stage2_avoid_vertical_half_height", stage2_avoid_vertical_half_height_, 1.6); // 第二赛段前向检测竖向半高（m）
    nh->param("stage2_avoid_center_half_width", stage2_avoid_center_half_width_, 0.9); // 第二赛段中心阻塞检测半宽（m）
    nh->param("stage2_avoid_side_probe_width", stage2_avoid_side_probe_width_, 2.5); // 第二赛段左右探测宽度（m）
    nh->param("stage2_avoid_max_lateral_offset", stage2_avoid_max_lateral_offset_, 2.0); // 第二赛段最大横向绕障偏置（m）
    nh->param("stage2_avoid_candidate_count", stage2_avoid_candidate_count_, 7); // 第二赛段候选绕障方向数量
    nh->param("stage2_avoid_ray_radius", stage2_avoid_ray_radius_, 0.9); // 第二赛段候选路径碰撞检测半径（m）
    nh->param("stage2_blockage_min_points", stage2_blockage_min_points_, 6); // 第二赛段判定“前方阻塞”的最少点数
    nh->param("stage2_blockage_fov_deg", stage2_blockage_fov_deg_, 60.0); // 第二赛段前向阻塞判定的视场角（deg）
    nh->param("stage2_blockage_vertical_fov_deg", stage2_blockage_vertical_fov_deg_, 45.0); // 第二赛段前向阻塞判定的垂直视场角（deg）
    nh->param("stage2_blockage_density_thresh", stage2_blockage_density_thresh_, 0.18); // 第二赛段前向阻塞判定的扇区占用密度阈值
    nh->param("stage2_blockage_sector_bins", stage2_blockage_sector_bins_, 9); // 第二赛段前向阻塞判定的角度分桶数
    nh->param("stage2_blockage_vertical_bins", stage2_blockage_vertical_bins_, 5); // 第二赛段前向阻塞判定的垂直分桶数
    nh->param("stage2_avoid_clearance_weight", stage2_avoid_clearance_weight_, 1.6); // 第二赛段候选评分：净空权重
    nh->param("stage2_avoid_alignment_weight", stage2_avoid_alignment_weight_, 0.9); // 第二赛段候选评分：朝前权重
    nh->param("stage2_avoid_offset_weight", stage2_avoid_offset_weight_, 0.7); // 第二赛段候选评分：偏置惩罚权重
    nh->param("stage2_avoid_density_weight", stage2_avoid_density_weight_, 0.6); // 第二赛段候选评分：扇区密度权重
    nh->param("stage2_avoid_density_sector_bins", stage2_avoid_density_sector_bins_, 5); // 第二赛段候选评分：候选扇区角度分桶数
    nh->param("stage2_avoid_density_vertical_bins", stage2_avoid_density_vertical_bins_, 3); // 第二赛段候选评分：候选扇区垂直分桶数
    nh->param("stage2_gap_safe_width", stage2_gap_safe_width_, 0.15); // 第二赛段候选通道需要满足的最小缝隙宽度（m）
    nh->param("stage2_gap_width_weight", stage2_gap_width_weight_, 1.2); // 第二赛段候选评分：缝隙宽度权重
    nh->param("stage2_corridor_width", stage2_corridor_width_, 10.0); // 第二赛段赛道通道宽度（m）
    nh->param("stage2_corridor_height", stage2_corridor_height_, 5.0); // 第二赛段赛道通道高度（m）
    // [P0-边界约束] 横向回拉参数
    nh->param("stage2_corridor_pull_soft_ratio", stage2_corridor_pull_soft_ratio_, 0.6);
    nh->param("stage2_corridor_pull_gain", stage2_corridor_pull_gain_, 1.5);
    nh->param("stage2_corridor_pull_max_vy", stage2_corridor_pull_max_vy_, 6.0);
    nh->param("stage2_corridor_hard_vx_scale", stage2_corridor_hard_vx_scale_, 0.4);
    nh->param("stage2_vehicle_width", stage2_vehicle_width_, 0.40); // 无人机宽度（m）
    nh->param("stage2_vehicle_height", stage2_vehicle_height_, 0.30); // 无人机高度（m）
    nh->param("stage2_vehicle_clearance", stage2_vehicle_clearance_, 0.15); // 无人机穿缝时的单侧安全余量（m）
    nh->param("stage2_occupancy_grid_fov_deg", stage2_occupancy_grid_fov_deg_, 140.0); // 第二赛段方向空间表示：水平总视场角（deg）
    nh->param("stage2_occupancy_grid_vfov_deg", stage2_occupancy_grid_vfov_deg_, 70.0); // 第二赛段方向空间表示：垂直总视场角（deg）
    nh->param("stage2_occupancy_grid_yaw_bins", stage2_occupancy_grid_yaw_bins_, 19); // 第二赛段方向空间表示：水平分桶数
    nh->param("stage2_occupancy_grid_pitch_bins", stage2_occupancy_grid_pitch_bins_, 7); // 第二赛段方向空间表示：垂直分桶数
    nh->param("stage2_process_scale_factor", stage2_process_scale_factor_, 3); // 第二赛段点云预处理：细分方向网格倍率
    nh->param("stage2_process_min_points_per_cell", stage2_process_min_points_per_cell_, 1); // 第二赛段点云预处理：每个细分cell保留的最小点数阈值
    nh->param("stage2_process_max_age_sec", stage2_process_max_age_sec_, 0.30); // 第二赛段点云预处理：短时历史保留时长（s）
    nh->param("stage2_avoid_trigger_dist", stage2_avoid_trigger_dist_, 1.5); // 第二赛段触发绕障的最近障碍距离阈值（m）
    nh->param("stage2_avoid_slowdown_dist", stage2_avoid_slowdown_dist_, stage2_avoid_trigger_dist_); // 兼容旧参数，内部统一按trigger处理
    nh->param("stage2_avoid_stop_dist", stage2_avoid_stop_dist_, 4.0); // 兼容旧参数，当前不再用于前向减速控制
    nh->param("stage2_avoid_min_scale", stage2_avoid_min_scale_, 0.15); // 兼容旧参数，当前不再用于前向减速控制
    nh->param("stage2_recover_clearance", stage2_recover_clearance_, 5.0); // 第二赛段判定“完全脱离障碍”所需净空（m）
    nh->param("stage2_recover_hold_sec", stage2_recover_hold_sec_, 0.6); // 第二赛段脱离障碍后保持局部避障的最短时间（s）
    nh->param("stage2_dynamic_closing_speed_thresh", stage2_dynamic_closing_speed_thresh_, 1.2); // 第二赛段把迎面动态障碍视为高威胁的最小接近速度（m/s）
    nh->param("stage2_dynamic_ttc_thresh", stage2_dynamic_ttc_thresh_, 1.0); // 第二赛段迎面动态障碍的最小碰撞时间阈值（s）

    refresh_stage2_geometry_baseline();
    nh->param("stage2_local_traj_spacing", stage2_local_traj_spacing_, 0.8); // 第二赛段局部轨迹点间距（m）
    nh->param("stage2_local_traj_lookahead_dist", stage2_local_traj_lookahead_dist_, 2.0); // 第二赛段局部轨迹跟踪前视距离（m）
    nh->param("stage1_turn_guard_enable", stage1_turn_guard_enable_, true); // 第一赛段危险弯道保护
    nh->param("stage1_turn_guard_s_min", stage1_turn_guard_s_min_, 900.0); // 第一赛段危险弯道起始弧长（m）
    nh->param("stage1_turn_guard_s_max", stage1_turn_guard_s_max_, 980.0); // 第一赛段危险弯道结束弧长（m）
    nh->param("stage1_turn_guard_lookahead", stage1_turn_guard_lookahead_, 8.0); // 第一赛段危险弯道前视距离（m）
    nh->param("stage1_turn_guard_max_vx", stage1_turn_guard_max_vx_, 13.0); // 第一赛段危险弯道前向速度上限（m/s）
    nh->param("stage1_turn_guard_max_vy", stage1_turn_guard_max_vy_, 4.0); // 第一赛段危险弯道横向速度上限（m/s）
    nh->param("use_relative_path", use_relative_path_, false); // 是否把CSV首点对齐到当前起点（当前实现固定绝对坐标）
    nh->param("invert_csv_y", invert_csv_y_, false);       // 是否对CSV的Y轴取反（默认不再翻转，表格已对齐方向）
    nh->param("fp_kp_xy", fp_kp_xy_, 1.8);               // 兼容旧参数：平面P增益（默认同步为X轴）
    nh->param("fp_kp_x", fp_kp_x_, 1.8);                 // 机体系X轴位置误差增益
    nh->param("fp_kp_y", fp_kp_y_, 2.0);                 // 机体系Y轴位置误差增益
    nh->param("fp_y_vel_damp", fp_y_vel_damp_, 1.5);     // 机体系Y轴速度阻尼增益
    nh->param("fp_vy_lpf_alpha", fp_vy_lpf_alpha_, 0.40); // vy命令低通滤波系数(0~1)
    nh->param("lateral_slow_err_thresh", lateral_slow_err_thresh_, 6.5); // 横向偏差超过阈值才降前向速度（m）
    nh->param("late_lateral_err_thresh", late_lateral_err_thresh_, 20.0); // 飞行后段放宽的横向门控阈值（m）
    nh->param("gate_relax_time_sec", gate_relax_time_sec_, 80.0);        // 启用放宽阈值的时间点（起飞后秒）
    nh->param("lateral_slow_gain", lateral_slow_gain_, 0.22);             // 超阈值后的前向降速强度
    nh->param("lateral_slow_min_scale", lateral_slow_min_scale_, 0.70);   // 前向降速最小比例
    nh->param("z_lookahead_dist", z_lookahead_dist_, 0.2);// Z通道前视距离（m），越小越贴线
    nh->param("max_vz_down", max_vz_down_, 10.0);   // Z轴下降速度上限（m/s，正值）
    nh->param("z_pid_kp", z_pid_kp_, 5.2);             // Z-P增益
    nh->param("z_pid_ki", z_pid_ki_, 1.4);               // Z-I增益
    nh->param("z_pid_kd", z_pid_kd_, 0.8);           // Z-D增益
    nh->param("z_pid_i_limit", z_pid_i_limit_, 2.2);       // Z轴积分限幅
    nh->param("z_pid_d_lpf_alpha", z_pid_d_lpf_alpha_, 0.35); // Z轴速度误差低通系数
    nh->param("z_vel_lpf_alpha", z_vel_lpf_alpha_, 0.30);     // Z轴实测速度低通系数
    nh->param("z_ref_vel_lpf_alpha", z_ref_vel_lpf_alpha_, 0.35); // Z轴目标速度低通系数
    nh->param("z_ref_vel_ff_gain", z_ref_vel_ff_gain_, 2.2); // Z轴目标速度前馈增益
    nh->param("fp_kp_yaw", fp_kp_yaw_, 3.0);               // 航向P增益：yaw误差->yawRate（与yaw_tuner快档一致）
    nh->param("fp_kd_yaw_rate", fp_kd_yaw_rate_, 0.12);    // 航向D增益：抑制过冲（基于实测偏航角速度）
    nh->param("yaw_rate_lpf_alpha", yaw_rate_lpf_alpha_, 0.32); // 偏航角速度低通滤波系数(0~1)
    nh->param("yaw_soft_limit_deg", yaw_soft_limit_deg_, 80.0); // yawRate软限幅（deg/s，与yaw_tuner快档一致）
    nh->param("yaw_stop_tolerance_deg", yaw_stop_tolerance_deg_, 0.8); // 航向停止容差（deg）
    nh->param("yaw_min_cmd_deg", yaw_min_cmd_deg_, 4.5);   // 航向最小角速度（deg/s），避免末段太慢
    nh->param("enable_yaw_only_gate", enable_yaw_only_gate_, true); // 是否启用大角误差只转不走门控
    nh->param("yaw_only_deg", yaw_only_deg_, 30.0);        // 启用门控时的阈值（deg）
    nh->param("pose_gt_is_ned", pose_gt_is_ned_, true);    // pose_gt是否按NED输入（true会做NED->ZUP转换）
    nh->param("use_takeoff_service", use_takeoff_service_, false); // 是否调用起飞服务
    ros::NodeHandle pnh("~");
    pnh.param("viz_frame_id", viz_frame_id_, std::string("map"));
    pnh.param("publish_follow_tf", publish_follow_tf_, true);
    pnh.param("follow_frame_id", follow_frame_id_, std::string("drone_follow"));
    pnh.param("actual_append_dist", actual_append_dist_, 0.05);
    int actual_max_points_param = static_cast<int>(actual_max_points_);
    pnh.param("actual_max_points", actual_max_points_param, 3000);
    actual_max_points_ = static_cast<size_t>(std::max(100, actual_max_points_param));
    pnh.param("corridor_enable", corridor_enable_, true);
    pnh.param("corridor_voxel_leaf", corridor_voxel_leaf_, 0.25);
    pnh.param("corridor_min_range", corridor_min_range_, 0.6);
    pnh.param("corridor_max_range", corridor_max_range_, 45.0);
    pnh.param("corridor_pub_hz", corridor_pub_hz_, 2.0);
    int corridor_max_points_param = static_cast<int>(corridor_max_points_);
    pnh.param("corridor_max_points", corridor_max_points_param, 250000);
    corridor_max_points_ = static_cast<size_t>(std::max(1000, corridor_max_points_param));
    corridor_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    stage2_frame_cloud_body_.reset(new pcl::PointCloud<pcl::PointXYZI>());
    corridor_cloud_->reserve(corridor_max_points_);
    corridor_voxel_keys_.reserve(corridor_max_points_);

    // 第一赛段参数强制锁定为提交镜像版本，避免被参数服务器残留值污染。
    test_speed_xy = 12.0;
    test_yaw_rate_deg = 45.0;
    test_move_dist = 15.0;
    test_hover_sec = 0.0;
    buffer_dist_ = 10.0;
    buffer_vx_ = 16.0;
    buffer_vz_ = 0.8;
    buffer_max_climb_ = 0.9;
    buffer_hover_sec_ = 0.0;
    max_vx_ = 18.0;
    max_vy_ = 5.0;
    max_vz_ = 12.0;
    max_vz_start_zone_ = 2.6;
    downsample_dist_ = 0.4;
    lookahead_dist_ = 6.5;
    lookahead_min_dist_ = 6.0;
    lookahead_max_dist_ = 10.0;
    lookahead_time_gain_ = 0.25;
    lookahead_speed_ref_ = 10.0;
    reach_end_dist_ = 0.6;
    final_brake_start_dist_ = 16.0;
    final_brake_min_scale_ = 0.20;
    final_stop_before_end_dist_ = 8.0;
    final_turn_deg_ = 180.0;
    final_turn_rate_limit_deg_ = 35.0;
    final_turn_tolerance_deg_ = 2.0;
    stage1_turn_guard_enable_ = true;
    stage1_turn_guard_s_min_ = 900.0;
    stage1_turn_guard_s_max_ = 980.0;
    stage1_turn_guard_lookahead_ = 8.0;
    stage1_turn_guard_max_vx_ = 13.0;
    stage1_turn_guard_max_vy_ = 4.0;
    use_relative_path_ = false;
    invert_csv_y_ = false;
    fp_kp_xy_ = 1.8;
    fp_kp_x_ = 1.8;
    fp_kp_y_ = 2.0;
    fp_y_vel_damp_ = 1.5;
    fp_vy_lpf_alpha_ = 0.40;
    lateral_slow_err_thresh_ = 6.5;
    late_lateral_err_thresh_ = 20.0;
    gate_relax_time_sec_ = 80.0;
    lateral_slow_gain_ = 0.22;
    lateral_slow_min_scale_ = 0.70;
    z_lookahead_dist_ = 0.2;
    max_vz_down_ = 10.0;
    z_pid_kp_ = 5.2;
    z_pid_ki_ = 1.4;
    z_pid_kd_ = 0.8;
    z_pid_i_limit_ = 2.2;
    z_pid_d_lpf_alpha_ = 0.35;
    z_vel_lpf_alpha_ = 0.30;
    z_ref_vel_lpf_alpha_ = 0.35;
    z_ref_vel_ff_gain_ = 2.2;
    fp_kp_yaw_ = 3.0;
    fp_kd_yaw_rate_ = 0.12;
    yaw_rate_lpf_alpha_ = 0.32;
    yaw_soft_limit_deg_ = 80.0;
    yaw_stop_tolerance_deg_ = 0.8;
    yaw_min_cmd_deg_ = 4.5;
    enable_yaw_only_gate_ = true;
    yaw_only_deg_ = 30.0;

    nh->setParam("test_speed_xy", test_speed_xy);
    nh->setParam("test_yaw_rate_deg", test_yaw_rate_deg);
    nh->setParam("test_move_dist", test_move_dist);
    nh->setParam("test_hover_sec", test_hover_sec);
    nh->setParam("buffer_dist", buffer_dist_);
    nh->setParam("buffer_vx", buffer_vx_);
    nh->setParam("buffer_vz", buffer_vz_);
    nh->setParam("buffer_max_climb", buffer_max_climb_);
    nh->setParam("buffer_hover_sec", buffer_hover_sec_);
    nh->setParam("max_vx", max_vx_);
    nh->setParam("max_vy", max_vy_);
    nh->setParam("max_vz", max_vz_);
    nh->setParam("max_vz_start_zone", max_vz_start_zone_);
    nh->setParam("downsample_dist", downsample_dist_);
    nh->setParam("lookahead_dist", lookahead_dist_);
    nh->setParam("lookahead_min_dist", lookahead_min_dist_);
    nh->setParam("lookahead_max_dist", lookahead_max_dist_);
    nh->setParam("lookahead_time_gain", lookahead_time_gain_);
    nh->setParam("lookahead_speed_ref", lookahead_speed_ref_);
    nh->setParam("reach_end_dist", reach_end_dist_);
    nh->setParam("final_brake_start_dist", final_brake_start_dist_);
    nh->setParam("final_brake_min_scale", final_brake_min_scale_);
    nh->setParam("final_stop_before_end_dist", final_stop_before_end_dist_);
    nh->setParam("final_turn_deg", final_turn_deg_);
    nh->setParam("final_turn_rate_limit_deg", final_turn_rate_limit_deg_);
    nh->setParam("final_turn_tolerance_deg", final_turn_tolerance_deg_);
    nh->setParam("stage1_turn_guard_enable", stage1_turn_guard_enable_);
    nh->setParam("stage1_turn_guard_s_min", stage1_turn_guard_s_min_);
    nh->setParam("stage1_turn_guard_s_max", stage1_turn_guard_s_max_);
    nh->setParam("stage1_turn_guard_lookahead", stage1_turn_guard_lookahead_);
    nh->setParam("stage1_turn_guard_max_vx", stage1_turn_guard_max_vx_);
    nh->setParam("stage1_turn_guard_max_vy", stage1_turn_guard_max_vy_);
    nh->setParam("use_relative_path", use_relative_path_);
    nh->setParam("invert_csv_y", invert_csv_y_);
    nh->setParam("fp_kp_xy", fp_kp_xy_);
    nh->setParam("fp_kp_x", fp_kp_x_);
    nh->setParam("fp_kp_y", fp_kp_y_);
    nh->setParam("fp_y_vel_damp", fp_y_vel_damp_);
    nh->setParam("fp_vy_lpf_alpha", fp_vy_lpf_alpha_);
    nh->setParam("lateral_slow_err_thresh", lateral_slow_err_thresh_);
    nh->setParam("late_lateral_err_thresh", late_lateral_err_thresh_);
    nh->setParam("gate_relax_time_sec", gate_relax_time_sec_);
    nh->setParam("lateral_slow_gain", lateral_slow_gain_);
    nh->setParam("lateral_slow_min_scale", lateral_slow_min_scale_);
    nh->setParam("z_lookahead_dist", z_lookahead_dist_);
    nh->setParam("max_vz_down", max_vz_down_);
    nh->setParam("z_pid_kp", z_pid_kp_);
    nh->setParam("z_pid_ki", z_pid_ki_);
    nh->setParam("z_pid_kd", z_pid_kd_);
    nh->setParam("z_pid_i_limit", z_pid_i_limit_);
    nh->setParam("z_pid_d_lpf_alpha", z_pid_d_lpf_alpha_);
    nh->setParam("z_vel_lpf_alpha", z_vel_lpf_alpha_);
    nh->setParam("z_ref_vel_lpf_alpha", z_ref_vel_lpf_alpha_);
    nh->setParam("z_ref_vel_ff_gain", z_ref_vel_ff_gain_);
    nh->setParam("fp_kp_yaw", fp_kp_yaw_);
    nh->setParam("fp_kd_yaw_rate", fp_kd_yaw_rate_);
    nh->setParam("yaw_rate_lpf_alpha", yaw_rate_lpf_alpha_);
    nh->setParam("yaw_soft_limit_deg", yaw_soft_limit_deg_);
    nh->setParam("yaw_stop_tolerance_deg", yaw_stop_tolerance_deg_);
    nh->setParam("yaw_min_cmd_deg", yaw_min_cmd_deg_);
    nh->setParam("enable_yaw_only_gate", enable_yaw_only_gate_);
    nh->setParam("yaw_only_deg", yaw_only_deg_);
    ROS_INFO("Stage1 params pinned to packaged baseline to avoid rosparam contamination.");

    // 参数限幅保护（保留物理合理范围，不再做“下限强推快档”）
    fp_kd_yaw_rate_ = std::max(fp_kd_yaw_rate_, 0.0);
    yaw_rate_lpf_alpha_ = std::max(0.0, std::min(1.0, yaw_rate_lpf_alpha_));
    yaw_stop_tolerance_deg_ = std::max(yaw_stop_tolerance_deg_, 0.0);
    yaw_min_cmd_deg_ = std::max(yaw_min_cmd_deg_, 0.0);
    // 兼容旧字段：统一把fp_kp_xy同步为X轴增益，防止旧逻辑读取到低值
    fp_kp_xy_ = fp_kp_x_;

    ROS_WARN("Z参数(启动生效): z_kp=%.3f z_ki=%.3f z_kd=%.3f z_i_lim=%.3f z_ff=%.3f z_lh=%.3f max_vz=%.3f max_vz_down=%.3f",
             z_pid_kp_, z_pid_ki_, z_pid_kd_, z_pid_i_limit_,
             z_ref_vel_ff_gain_, z_lookahead_dist_, max_vz_, max_vz_down_);

    ROS_INFO("basic_dev 已启动：起飞后进入CSV航迹跟踪模式（invert_csv_y=%d, pose_gt_is_ned=%d）。",
        static_cast<int>(invert_csv_y_), static_cast<int>(pose_gt_is_ned_));
    ros::spin();
}

BasicDev::~BasicDev()
{
}

// -----------------------------------------------------------------------------
// Small utilities and phase bookkeeping
// -----------------------------------------------------------------------------
double BasicDev::clamp_value(double value, double lower, double upper)
{
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}

const char* BasicDev::phase_name(TestPhase p) const
{
    switch (p) {
    case TestPhase::WAITING_DATA: return "WAITING_DATA";
    case TestPhase::HOVER_AFTER_TAKEOFF: return "HOVER_AFTER_TAKEOFF";
    case TestPhase::MOVE_FORWARD_BUFFER: return "MOVE_FORWARD_BUFFER";
    case TestPhase::HOVER_AFTER_BUFFER: return "HOVER_AFTER_BUFFER";
    case TestPhase::FOLLOW_PATH: return "FOLLOW_PATH";
    case TestPhase::FINAL_HOVER_TURN: return "FINAL_HOVER_TURN";
    case TestPhase::WAIT_STAGE2_MANUAL: return "WAIT_STAGE2_MANUAL";
    case TestPhase::FOLLOW_PATH_STAGE2: return "FOLLOW_PATH_STAGE2";
    case TestPhase::MOVE_FORWARD: return "MOVE_FORWARD";
    case TestPhase::MOVE_BACKWARD: return "MOVE_BACKWARD";
    case TestPhase::MOVE_RIGHT: return "MOVE_RIGHT";
    case TestPhase::MOVE_LEFT: return "MOVE_LEFT";
    case TestPhase::YAW_CW: return "YAW_CW";
    case TestPhase::YAW_CCW: return "YAW_CCW";
    case TestPhase::FINISH: return "FINISH";
    default: return "UNKNOWN";
    }
}

const char* BasicDev::stage2_mode_name(Stage2FollowMode m) const
{
    switch (m) {
    case Stage2FollowMode::CRUISE: return "CRUISE";
    case Stage2FollowMode::AVOID: return "AVOID";
    default: return "UNKNOWN";
    }
}

void BasicDev::refresh_stage2_geometry_baseline()
{
    const double clearance = std::max(0.0, stage2_vehicle_clearance_);
    stage2_vehicle_safe_width_ = std::max(0.2, stage2_vehicle_width_ + 2.0 * clearance);
    stage2_vehicle_safe_height_ = std::max(0.2, stage2_vehicle_height_ + 2.0 * clearance);
    stage2_corridor_half_width_safe_ =
        std::max(0.5, 0.5 * stage2_corridor_width_ - 0.5 * stage2_vehicle_width_ - clearance);
    stage2_corridor_half_height_safe_ =
        std::max(0.3, 0.5 * stage2_corridor_height_ - 0.5 * stage2_vehicle_height_ - clearance);
    stage2_gap_safe_width_ = std::max(stage2_gap_safe_width_, stage2_vehicle_safe_width_);
    stage2_avoid_max_lateral_offset_ =
        std::min(stage2_avoid_max_lateral_offset_, stage2_corridor_half_width_safe_);
    stage2_dynamic_avoid_max_lateral_offset_ =
        std::min(stage2_dynamic_avoid_max_lateral_offset_, stage2_corridor_half_width_safe_);
}

void BasicDev::enter_phase(TestPhase p, const ros::Time& now)
{
    phase = p;
    phase_start_time = now;
    if (nh_) {
        nh_->setParam("/basic_dev/current_phase", phase_name(phase));
    }
    if (phase == TestPhase::FOLLOW_PATH || phase == TestPhase::FOLLOW_PATH_STAGE2) {
        // 切入路径跟踪时重置Z轴PID，避免前一阶段积分/微分残留
        z_pid_int_ = 0.0;
        z_pid_prev_err_ = 0.0;
        z_pid_d_filt_ = 0.0;
        z_pid_inited_ = false;
        z_vel_inited_ = false;
        z_prev_meas_ = 0.0;
        z_prev_ref_ = 0.0;
        z_vel_meas_filt_ = 0.0;
        z_ref_vel_filt_ = 0.0;
        yaw_rate_inited_ = false;
        yaw_prev_rad_ = 0.0;
        yaw_rate_meas_filt_ = 0.0;
        xy_speed_inited_ = false;
    }
    if (phase == TestPhase::FINAL_HOVER_TURN) {
        final_turn_inited_ = false;
    }
    if (phase == TestPhase::WAIT_STAGE2_MANUAL) {
        stage2_manual_zero_sent_ = false;
        stage2_mode_ = Stage2FollowMode::CRUISE;
        if (nh_) nh_->setParam("/basic_dev/stage2_manual_takeover_ready", true);
    } else if (nh_) {
        nh_->setParam("/basic_dev/stage2_manual_takeover_ready", false);
    }
    if (phase == TestPhase::FOLLOW_PATH_STAGE2) {
        stage2_mode_ = Stage2FollowMode::CRUISE;
        stage2_cmd_inited_ = false;
        stage2_prev_vy_cmd_ = 0.0;
        stage2_prev_vz_cmd_ = 0.0;
        stage2_avoid_latched_ = false;
        stage2_triggered_streak_ = 0;
        stage2_last_avoid_target_valid_ = false;
        stage2_smoothed_avoid_target_valid_ = false;
        stage2_last_candidate_yaw_idx_ = -1;
        stage2_last_candidate_pitch_idx_ = -1;
        stage2_local_traj_.clear();
        stage2_last_blocked_time_ = ros::Time(0);
        stage2_prev_blockage_dist_ = std::numeric_limits<double>::infinity();
        stage2_prev_blockage_time_ = ros::Time(0);
        stage2_z_blend_pause_xy_ = 0.0;
        stage2_prev_travel_xy_ = 0.0;
    }
    ROS_INFO("进入阶段: %s", phase_name(phase));
}

double BasicDev::phase_elapsed(const ros::Time& now) const
{
    return (now - phase_start_time).toSec();
}

static inline double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

static inline double wrap_pi_rad(double rad)
{
    while (rad > M_PI) rad -= 2.0 * M_PI;
    while (rad < -M_PI) rad += 2.0 * M_PI;
    return rad;
}

static inline double wrap_deg(double deg)
{
    while (deg > 180.0) deg -= 360.0;
    while (deg < -180.0) deg += 360.0;
    return deg;
}

static inline double quat_to_yaw_rad(const Eigen::Quaterniond& q)
{
    const Eigen::Quaterniond qn = q.normalized();
    const double siny_cosp = 2.0 * (qn.w() * qn.z() + qn.x() * qn.y());
    const double cosy_cosp = 1.0 - 2.0 * (qn.y() * qn.y() + qn.z() * qn.z());
    return std::atan2(siny_cosp, cosy_cosp);
}

static inline Eigen::Quaterniond quat_ned_to_zup(const Eigen::Quaterniond& q_ned)
{
    // Internal convention: x forward, y right, z up.
    // AirSim GPS pose uses z-down convention, so convert rotation by reflecting z axis.
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    S(2, 2) = -1.0;
    const Eigen::Matrix3d R_int = S * q_ned.normalized().toRotationMatrix() * S;
    return Eigen::Quaterniond(R_int).normalized();
}

static inline long long pack_voxel_key(int ix, int iy, int iz)
{
    // 3D 体素索引打包为 64 位键值（每轴21bit，带符号偏移）
    constexpr long long kBits = 21;
    constexpr long long kMask = (1LL << kBits) - 1LL;
    constexpr long long kOffset = 1LL << (kBits - 1LL);
    const long long x = (static_cast<long long>(ix) + kOffset) & kMask;
    const long long y = (static_cast<long long>(iy) + kOffset) & kMask;
    const long long z = (static_cast<long long>(iz) + kOffset) & kMask;
    return (x << (kBits * 2LL)) | (y << kBits) | z;
}

// -----------------------------------------------------------------------------
// Global path loading and indexing
// -----------------------------------------------------------------------------
bool BasicDev::load_path_from_csv(const Eigen::Vector3d& start_world)
{
    (void)start_world;
    std::ifstream fin(csv_path_);
    if (!fin.is_open()) {
        ROS_ERROR("Cannot open csv_path: %s", csv_path_.c_str());
        return false;
    }
    std::string line;
    std::getline(fin, line); // header

    std::vector<Eigen::Vector3d> raw;
    raw.reserve(6000);

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        const auto pos = line.find_first_not_of(" \t\r");
        if (pos == std::string::npos) continue;
        if (line[pos] == '#') continue;

        std::stringstream ss(line);
        std::string tok;
        std::vector<double> vals;
        try {
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) vals.push_back(std::stod(tok));
            }
        } catch (...) {
            continue;
        }
        if (vals.size() < 4) continue; // t,x,y,z

        const double x = vals[1];
        double y = vals[2];
        if (invert_csv_y_) {
            y = -y;
        }
        const double z = vals[3];
        raw.emplace_back(x, y, z);
    }

    if (raw.size() < 2) {
        ROS_ERROR("CSV points too few: %zu", raw.size());
        return false;
    }

    // 按CSV绝对坐标直接使用，不做任何平移对齐
    if (use_relative_path_) {
        ROS_WARN_THROTTLE(2.0, "use_relative_path=true 已被忽略：当前固定使用CSV绝对坐标。");
    }

    // downsample by distance
    std::vector<Eigen::Vector3d> ds;
    ds.reserve(raw.size());
    ds.push_back(raw.front());
    for (size_t i = 1; i < raw.size(); ++i) {
        if ((raw[i] - ds.back()).norm() >= downsample_dist_)
            ds.push_back(raw[i]);
    }
    if ((ds.back() - raw.back()).norm() > 1e-6) ds.push_back(raw.back());

    desired_path_msg_.poses.clear();
    stage2_active_ = false;
    stage2_start_pos_valid_ = false;
    path_pts_.swap(ds);
    path_s_.clear();
    path_s_.reserve(path_pts_.size());
    double acc_s = 0.0;
    path_s_.push_back(0.0);
    for (size_t i = 1; i < path_pts_.size(); ++i) {
        acc_s += (path_pts_[i] - path_pts_[i - 1]).norm();
        path_s_.push_back(acc_s);
    }
    path_loaded_ = true;
    publish_desired_path_once();

    ROS_INFO("Path loaded: raw=%zu downsample=%zu first=(%.2f %.2f %.2f) last=(%.2f %.2f %.2f)",
             raw.size(), path_pts_.size(),
             path_pts_.front().x(), path_pts_.front().y(), path_pts_.front().z(),
             path_pts_.back().x(),  path_pts_.back().y(),  path_pts_.back().z());
    return true;
}

bool BasicDev::load_stage2_path_from_anchor(const Eigen::Vector3d& anchor_world)
{
    std::ifstream fin(stage2_csv_path_);
    if (!fin.is_open()) {
        ROS_ERROR("Cannot open stage2_csv_path: %s", stage2_csv_path_.c_str());
        return false;
    }

    std::string line;
    std::getline(fin, line); // header

    std::vector<Eigen::Vector3d> raw;
    raw.reserve(2000);
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        const auto pos = line.find_first_not_of(" \t\r");
        if (pos == std::string::npos) continue;
        if (line[pos] == '#') continue;

        std::stringstream ss(line);
        std::string tok;
        std::vector<double> vals;
        try {
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) vals.push_back(std::stod(tok));
            }
        } catch (...) {
            continue;
        }
        if (vals.size() < 4) continue;

        const double x = vals[1];
        double y = vals[2];
        if (stage2_invert_csv_y_) {
            y = -y;
        }
        const double z = vals[3];
        raw.emplace_back(x, y, z);
    }

    if (raw.size() < 2) {
        ROS_ERROR("Stage2 CSV points too few: %zu", raw.size());
        return false;
    }

    size_t nearest = 0;
    double best_d2 = 1e18;
    for (size_t i = 0; i < raw.size(); ++i) {
        const double d2 = (raw[i] - anchor_world).squaredNorm();
        if (d2 < best_d2) {
            best_d2 = d2;
            nearest = i;
        }
    }

    if (nearest + 1 >= raw.size()) {
        ROS_ERROR("Stage2 anchor is too close to the end of the path: idx=%zu size=%zu", nearest, raw.size());
        return false;
    }

    std::vector<Eigen::Vector3d> trimmed(raw.begin() + static_cast<long>(nearest), raw.end());
    trimmed.front() = anchor_world;

    std::vector<Eigen::Vector3d> ds;
    ds.reserve(trimmed.size());
    ds.push_back(trimmed.front());
    for (size_t i = 1; i < trimmed.size(); ++i) {
        if ((trimmed[i] - ds.back()).norm() >= downsample_dist_) {
            ds.push_back(trimmed[i]);
        }
    }
    if ((ds.back() - trimmed.back()).norm() > 1e-6) ds.push_back(trimmed.back());

    if (ds.size() < 2) {
        ROS_ERROR("Stage2 path too short after anchor trim: %zu", ds.size());
        return false;
    }

    desired_path_msg_.poses.clear();
    path_pts_.swap(ds);
    path_s_.clear();
    path_s_.reserve(path_pts_.size());
    double acc_s = 0.0;
    path_s_.push_back(0.0);
    for (size_t i = 1; i < path_pts_.size(); ++i) {
        acc_s += (path_pts_[i] - path_pts_[i - 1]).norm();
        path_s_.push_back(acc_s);
    }
    path_loaded_ = true;
    stage2_active_ = true;
    stage2_start_pos_ = anchor_world;
    stage2_start_pos_valid_ = true;
    publish_desired_path_once();

    ROS_INFO("Stage2 path loaded: raw=%zu trim_idx=%zu kept=%zu first=(%.2f %.2f %.2f) last=(%.2f %.2f %.2f) anchor_err=%.2f",
             raw.size(), nearest, path_pts_.size(),
             path_pts_.front().x(), path_pts_.front().y(), path_pts_.front().z(),
             path_pts_.back().x(), path_pts_.back().y(), path_pts_.back().z(),
             std::sqrt(best_d2));
    return true;
}

size_t BasicDev::find_nearest_index(const Eigen::Vector3d& p) const
{
    size_t best = 0;
    double best_d2 = 1e18;
    for (size_t i = 0; i < path_pts_.size(); ++i) {
        const double d2 = (path_pts_[i] - p).squaredNorm();
        if (d2 < best_d2) { best_d2 = d2; best = i; }
    }
    return best;
}

size_t BasicDev::find_lookahead_index(size_t i_near, double L) const
{
    double acc = 0.0;
    for (size_t i = i_near; i + 1 < path_pts_.size(); ++i) {
        acc += (path_pts_[i + 1] - path_pts_[i]).norm();
        if (acc >= L) return i + 1;
    }
    return path_pts_.size() - 1;
}

static size_t find_nearest_index_in_points(const std::vector<Eigen::Vector3d>& pts, const Eigen::Vector3d& p)
{
    size_t best = 0;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < pts.size(); ++i) {
        const double d2 = (pts[i] - p).squaredNorm();
        if (d2 < best_d2) {
            best_d2 = d2;
            best = i;
        }
    }
    return best;
}

static size_t find_lookahead_index_in_points(const std::vector<Eigen::Vector3d>& pts, size_t i_near, double L)
{
    double acc = 0.0;
    for (size_t i = i_near; i + 1 < pts.size(); ++i) {
        acc += (pts[i + 1] - pts[i]).norm();
        if (acc >= L) return i + 1;
    }
    return pts.empty() ? 0 : (pts.size() - 1);
}

// -----------------------------------------------------------------------------
// Stage2 perception: current-frame point cloud -> direction-space occupancy
// -----------------------------------------------------------------------------
bool BasicDev::build_stage2_occupancy_grid(const Eigen::Vector3d& raw_target,
                                           Stage2OccupancyGrid& grid) const
{
    if (!stage2_active_ || !stage2_avoid_enable_ || !stage2_frame_cloud_body_ || stage2_frame_cloud_body_->points.empty()) {
        return false;
    }

    const Eigen::Matrix3d R_wb = current_quat.normalized().toRotationMatrix();
    const Eigen::Vector3d raw_target_b = R_wb.transpose() * (raw_target - current_pos);
    const double probe_len = clampd(raw_target_b.head<2>().norm(), 3.0, std::max(0.5, stage2_avoid_forward_range_));
    const double max_offset = std::max(0.0, stage2_avoid_max_lateral_offset_);
    const double center_y_lim = std::max(0.2, stage2_avoid_center_half_width_);
    const double goal_yaw = std::atan2(raw_target_b.y(), std::max(0.1, raw_target_b.x()));
    double last_avoid_yaw = goal_yaw;
    bool have_last_avoid_yaw = false;
    if (stage2_last_avoid_target_valid_) {
        const Eigen::Vector3d last_target_b = R_wb.transpose() * (stage2_last_avoid_target_ - current_pos);
        if (last_target_b.head<2>().norm() > 0.5) {
            last_avoid_yaw = std::atan2(last_target_b.y(), std::max(0.1, last_target_b.x()));
            have_last_avoid_yaw = true;
        }
    }
    const double max_deflect =
        clampd(std::atan2(std::max(max_offset, center_y_lim), probe_len) + 0.30, 0.45, 1.35);
    const double desired_half_yaw =
        std::max(0.5 * stage2_occupancy_grid_fov_deg_ * M_PI / 180.0,
                 std::fabs(goal_yaw) + max_deflect + 0.20);
    grid.yaw_bins = std::max(5, stage2_occupancy_grid_yaw_bins_ | 1);
    grid.pitch_bins = std::max(3, stage2_occupancy_grid_pitch_bins_ | 1);
    grid.yaw_min = -clampd(desired_half_yaw, 0.35, 1.45);
    grid.yaw_max = -grid.yaw_min;
    const double desired_half_pitch = 0.5 * stage2_occupancy_grid_vfov_deg_ * M_PI / 180.0;
    grid.pitch_min = -clampd(desired_half_pitch, 0.20, 1.20);
    grid.pitch_max = -grid.pitch_min;
    grid.max_range = std::max(0.5, stage2_avoid_forward_range_);
    grid.total_points = 0;
    grid.point_counts.assign(static_cast<size_t>(grid.yaw_bins * grid.pitch_bins), 0);
    grid.min_ranges.assign(static_cast<size_t>(grid.yaw_bins * grid.pitch_bins), std::numeric_limits<double>::infinity());
    grid.mean_ranges.assign(static_cast<size_t>(grid.yaw_bins * grid.pitch_bins), 0.0);
    grid.cell_costs.assign(static_cast<size_t>(grid.yaw_bins * grid.pitch_bins), 1.0);
    std::vector<double> range_sums(static_cast<size_t>(grid.yaw_bins * grid.pitch_bins), 0.0);

    const double corridor_half_height_safe =
        std::max(0.3,
                 stage2_corridor_half_height_safe_);
    const double z_lim = std::max(stage2_avoid_vertical_half_height_, corridor_half_height_safe);
    for (const auto& p : stage2_frame_cloud_body_->points) {
        const Eigen::Vector3d p_b(p.x, p.y, p.z);
        if (p_b.x() <= 0.0) continue;
        if (std::fabs(p_b.z()) > z_lim) continue;
        const Eigen::Vector2d p_xy = p_b.head<2>();
        const double xy_range = p_xy.norm();
        if (xy_range > grid.max_range) continue;
        const double bearing = std::atan2(p_b.y(), std::max(1e-3, p_b.x()));
        if (bearing < grid.yaw_min || bearing > grid.yaw_max) continue;
        const double elev = std::atan2(p_b.z(), std::max(1e-3, p_xy.norm()));
        if (elev < grid.pitch_min || elev > grid.pitch_max) continue;
        const double yaw_ratio = (bearing - grid.yaw_min) / std::max(1e-3, grid.yaw_max - grid.yaw_min);
        const double pitch_ratio = (elev - grid.pitch_min) / std::max(1e-3, grid.pitch_max - grid.pitch_min);
        const int yaw_bin = std::max(0, std::min(grid.yaw_bins - 1,
            static_cast<int>(std::floor(yaw_ratio * static_cast<double>(grid.yaw_bins)))));
        const int pitch_bin = std::max(0, std::min(grid.pitch_bins - 1,
            static_cast<int>(std::floor(pitch_ratio * static_cast<double>(grid.pitch_bins)))));
        const size_t flat = static_cast<size_t>(pitch_bin * grid.yaw_bins + yaw_bin);
        ++grid.point_counts[flat];
        grid.min_ranges[flat] = std::min(grid.min_ranges[flat], xy_range);
        range_sums[flat] += xy_range;
        ++grid.total_points;
    }

    if (grid.total_points <= 0) {
        return false;
    }

    for (size_t i = 0; i < grid.point_counts.size(); ++i) {
        if (grid.point_counts[i] > 0) {
            grid.mean_ranges[i] = range_sums[i] / static_cast<double>(grid.point_counts[i]);
        } else {
            grid.mean_ranges[i] = grid.max_range;
        }
    }

    std::vector<double> base_costs(grid.point_counts.size(), 1.0);
    for (size_t i = 0; i < grid.point_counts.size(); ++i) {
        if (grid.point_counts[i] <= 0) {
            base_costs[i] = 0.0;
            continue;
        }
        const double occ_conf = clampd(static_cast<double>(grid.point_counts[i]) /
                                           static_cast<double>(std::max(1, stage2_process_min_points_per_cell_ + 1)),
                                       0.15, 1.0);
        const double mean_norm = 1.0 - clampd(grid.mean_ranges[i] / std::max(1e-3, grid.max_range), 0.0, 1.0);
        const double near_norm = 1.0 - clampd(grid.min_ranges[i] / std::max(1e-3, grid.max_range), 0.0, 1.0);
        base_costs[i] = clampd(0.45 * occ_conf + 0.25 * mean_norm + 0.30 * near_norm, 0.0, 1.0);
    }

    const int smooth_radius_yaw = std::max(1, stage2_process_scale_factor_ / 2);
    const int smooth_radius_pitch = 1;
    for (int p = 0; p < grid.pitch_bins; ++p) {
        for (int y = 0; y < grid.yaw_bins; ++y) {
            double accum = 0.0;
            int samples = 0;
            for (int dp = -smooth_radius_pitch; dp <= smooth_radius_pitch; ++dp) {
                const int pp = p + dp;
                if (pp < 0 || pp >= grid.pitch_bins) continue;
                for (int dy = -smooth_radius_yaw; dy <= smooth_radius_yaw; ++dy) {
                    int yy = y + dy;
                    while (yy < 0) yy += grid.yaw_bins;
                    while (yy >= grid.yaw_bins) yy -= grid.yaw_bins;
                    accum += base_costs[static_cast<size_t>(pp * grid.yaw_bins + yy)];
                    ++samples;
                }
            }
            grid.cell_costs[static_cast<size_t>(p * grid.yaw_bins + y)] =
                (samples > 0) ? accum / static_cast<double>(samples) : 1.0;
        }
    }

    return true;
}

bool BasicDev::compute_stage2_blockage_metrics(const Stage2OccupancyGrid& grid,
                                               double& nearest_along,
                                               int& hit_count,
                                               double& density,
                                               int& occupied_bins) const
{
    nearest_along = std::numeric_limits<double>::infinity();
    hit_count = 0;
    density = 0.0;
    occupied_bins = 0;
    if (grid.yaw_bins <= 0 || grid.pitch_bins <= 0 || grid.total_points <= 0) {
        return false;
    }

    const double half_blockage_yaw = clampd(stage2_blockage_fov_deg_ * M_PI / 180.0 * 0.5, 0.10, 1.45);
    const double half_blockage_pitch = clampd(stage2_blockage_vertical_fov_deg_ * M_PI / 180.0 * 0.5, 0.08, 1.20);

    for (int pitch_bin = 0; pitch_bin < grid.pitch_bins; ++pitch_bin) {
        const double pitch_center = grid.pitch_min +
            (static_cast<double>(pitch_bin) + 0.5) * (grid.pitch_max - grid.pitch_min) / static_cast<double>(grid.pitch_bins);
        if (std::fabs(pitch_center) > half_blockage_pitch) continue;
        for (int yaw_bin = 0; yaw_bin < grid.yaw_bins; ++yaw_bin) {
            const double yaw_center = grid.yaw_min +
                (static_cast<double>(yaw_bin) + 0.5) * (grid.yaw_max - grid.yaw_min) / static_cast<double>(grid.yaw_bins);
            if (std::fabs(yaw_center) > half_blockage_yaw) continue;
            const size_t flat = static_cast<size_t>(pitch_bin * grid.yaw_bins + yaw_bin);
            const int count = grid.point_counts[flat];
            if (count <= 0) continue;
            ++occupied_bins;
            hit_count += count;
            nearest_along = std::min(nearest_along, grid.min_ranges[flat]);
        }
    }

    const int total_block_bins =
        std::max(1, static_cast<int>(std::round((2.0 * half_blockage_yaw) / std::max(1e-3, grid.yaw_max - grid.yaw_min) * grid.yaw_bins))) *
        std::max(1, static_cast<int>(std::round((2.0 * half_blockage_pitch) / std::max(1e-3, grid.pitch_max - grid.pitch_min) * grid.pitch_bins)));
    density = static_cast<double>(occupied_bins) / static_cast<double>(std::max(1, total_block_bins));

    return (hit_count >= stage2_blockage_min_points_) &&
           (density >= stage2_blockage_density_thresh_) &&
           std::isfinite(nearest_along) &&
           (nearest_along < stage2_avoid_trigger_dist_);
}

bool BasicDev::compute_stage2_avoid_target(const Eigen::Vector3d& raw_target,
                                           const Stage2OccupancyGrid& grid,
                                           Eigen::Vector3d& avoid_target,
                                           double& avoid_offset)
{
    avoid_target = raw_target;
    avoid_offset = 0.0;
    stage2_debug_candidate_targets_.clear();
    stage2_debug_candidate_scores_.clear();
    stage2_debug_candidate_clearances_.clear();
    stage2_debug_candidate_gap_widths_.clear();
    stage2_debug_selected_candidate_ = -1;
    stage2_debug_best_candidate_clearance_ = std::numeric_limits<double>::infinity();
    stage2_debug_best_gap_width_ = 0.0;

    if (!stage2_active_ || !stage2_avoid_enable_ || grid.total_points <= 0) {
        return false;
    }

    const Eigen::Matrix3d R_wb = current_quat.normalized().toRotationMatrix();
    const Eigen::Vector3d raw_target_b = R_wb.transpose() * (raw_target - current_pos);
    double last_avoid_yaw = std::atan2(raw_target_b.y(), std::max(0.1, raw_target_b.x()));
    bool have_last_avoid_yaw = false;
    if (stage2_last_avoid_target_valid_) {
        const Eigen::Vector3d last_target_b = R_wb.transpose() * (stage2_last_avoid_target_ - current_pos);
        if (last_target_b.head<2>().norm() > 0.5) {
            last_avoid_yaw = std::atan2(last_target_b.y(), std::max(0.1, last_target_b.x()));
            have_last_avoid_yaw = true;
        }
    }
    const size_t raw_target_idx = find_nearest_index(raw_target);
    const size_t path_prev_idx = (raw_target_idx > 0) ? raw_target_idx - 1 : raw_target_idx;
    const size_t path_next_idx = std::min(path_pts_.size() - 1, raw_target_idx + 1);
    Eigen::Vector3d path_tangent_w = path_pts_[path_next_idx] - path_pts_[path_prev_idx];
    path_tangent_w.z() = 0.0;
    if (path_tangent_w.head<2>().norm() < 1e-3) {
        path_tangent_w = R_wb.col(0);
        path_tangent_w.z() = 0.0;
    }
    if (path_tangent_w.head<2>().norm() < 1e-3) {
        path_tangent_w = Eigen::Vector3d::UnitX();
    } else {
        path_tangent_w.normalize();
    }
    Eigen::Vector3d path_lateral_w(-path_tangent_w.y(), path_tangent_w.x(), 0.0);
    if (path_lateral_w.head<2>().norm() < 1e-3) {
        path_lateral_w = Eigen::Vector3d::UnitY();
    } else {
        path_lateral_w.normalize();
    }
    const double center_y_lim = std::max(0.2, stage2_avoid_center_half_width_);
    const double max_offset = std::max(0.0, stage2_avoid_max_lateral_offset_);
    const int candidate_count = std::max(3, stage2_avoid_candidate_count_ | 1);

    const double goal_yaw = std::atan2(raw_target_b.y(), std::max(0.1, raw_target_b.x()));
    const double probe_len = clampd(raw_target_b.head<2>().norm(), 3.0, grid.max_range);
    const double max_deflect = clampd(std::atan2(std::max(max_offset, center_y_lim), probe_len) + 0.25, 0.35, 1.20);
    const double nearest_center_range = stage2_debug_blockage_dist_;

    const double activation = 1.0;

    stage2_debug_candidate_targets_.reserve(static_cast<size_t>(candidate_count));
    stage2_debug_candidate_scores_.reserve(static_cast<size_t>(candidate_count));
    stage2_debug_candidate_clearances_.reserve(static_cast<size_t>(candidate_count));
    stage2_debug_candidate_gap_widths_.reserve(static_cast<size_t>(candidate_count));

    const double candidate_pitch_limit = clampd(stage2_avoid_max_pitch_deg_ * M_PI / 180.0, 0.05, 0.5 * stage2_occupancy_grid_vfov_deg_ * M_PI / 180.0);
    const double yaw_step = (grid.yaw_max - grid.yaw_min) / std::max(1, grid.yaw_bins);
    const double pitch_step = (grid.pitch_max - grid.pitch_min) / std::max(1, grid.pitch_bins);
    const auto pitch_to_idx = [&](double pitch_rad) {
        const double pitch_ratio =
            (pitch_rad - grid.pitch_min) / std::max(1e-3, grid.pitch_max - grid.pitch_min);
        return std::max(0, std::min(grid.pitch_bins - 1,
            static_cast<int>(std::floor(pitch_ratio * static_cast<double>(grid.pitch_bins)))));
    };
    std::vector<int> allowed_pitch_bins = {
        pitch_to_idx(-0.65 * candidate_pitch_limit),
        pitch_to_idx(0.0),
        pitch_to_idx(0.65 * candidate_pitch_limit)
    };
    // Keep vertical choices intentionally small: down / level / up.
    std::sort(allowed_pitch_bins.begin(), allowed_pitch_bins.end());
    allowed_pitch_bins.erase(std::unique(allowed_pitch_bins.begin(), allowed_pitch_bins.end()), allowed_pitch_bins.end());
    Eigen::MatrixXf distance_cost(grid.pitch_bins, grid.yaw_bins);
    Eigen::MatrixXf other_cost(grid.pitch_bins, grid.yaw_bins);
    distance_cost.fill(0.0f);
    other_cost.fill(0.0f);
    const double distance_bias = std::max(0.5, 0.6 * stage2_avoid_trigger_dist_);
    const double corridor_half_width_safe =
        std::max(0.5,
                 stage2_corridor_half_width_safe_);
    const double corridor_half_height_safe =
        std::max(0.3,
                 stage2_corridor_half_height_safe_);
    const double soft_corridor_half_width = std::max(0.3, 0.75 * corridor_half_width_safe);
    const double soft_corridor_half_height = std::max(0.2, 0.75 * corridor_half_height_safe);
    const double clearance_radius =
        std::max(0.25, 0.5 * stage2_vehicle_safe_width_);
    const double z_clearance =
        std::max(0.20, 0.5 * stage2_vehicle_safe_height_);
    const double safety_radius = std::sqrt(clearance_radius * clearance_radius + z_clearance * z_clearance);
    const auto estimate_gap_width = [&](int pitch_idx, int yaw_idx, double ref_range) {
        const double range_for_gap = std::max(1.0, ref_range);
        int left_open = 0;
        for (int y = yaw_idx - 1; y >= 0; --y) {
            const size_t flat = static_cast<size_t>(pitch_idx * grid.yaw_bins + y);
            if (grid.point_counts[flat] > 0 || grid.cell_costs[flat] > 0.35) break;
            ++left_open;
        }
        int right_open = 0;
        for (int y = yaw_idx + 1; y < grid.yaw_bins; ++y) {
            const size_t flat = static_cast<size_t>(pitch_idx * grid.yaw_bins + y);
            if (grid.point_counts[flat] > 0 || grid.cell_costs[flat] > 0.35) break;
            ++right_open;
        }
        const int open_bins = left_open + right_open + 1;
        const double open_angle = static_cast<double>(open_bins) * yaw_step;
        return 2.0 * range_for_gap * std::tan(0.5 * std::max(0.0, open_angle));
    };
    const auto segment_collision_cost = [&](const Eigen::Vector3d& seg_start_b,
                                            const Eigen::Vector3d& seg_end_b) {
        if (!stage2_frame_cloud_body_ || stage2_frame_cloud_body_->points.empty()) return 0.0;
        const double x_min = std::min(seg_start_b.x(), seg_end_b.x()) - clearance_radius;
        const double x_max = std::max(seg_start_b.x(), seg_end_b.x()) + clearance_radius;
        const double y_min = std::min(seg_start_b.y(), seg_end_b.y()) - clearance_radius;
        const double y_max = std::max(seg_start_b.y(), seg_end_b.y()) + clearance_radius;
        const double z_min = std::min(seg_start_b.z(), seg_end_b.z()) - z_clearance;
        const double z_max = std::max(seg_start_b.z(), seg_end_b.z()) + z_clearance;
        int hit_count = 0;
        double nearest_dist = std::numeric_limits<double>::infinity();
        for (const auto& p : stage2_frame_cloud_body_->points) {
            if (p.x < x_min || p.x > x_max || p.y < y_min || p.y > y_max || p.z < z_min || p.z > z_max) continue;
            const Eigen::Vector3d p_b(p.x, p.y, p.z);
            const double dist = point_to_segment_distance(p_b, seg_start_b, seg_end_b);
            if (dist <= safety_radius) {
                ++hit_count;
                nearest_dist = std::min(nearest_dist, dist);
            }
        }
        if (hit_count <= 0) return 0.0;
        const double near_penalty = 1.0 - clampd(nearest_dist / std::max(1e-3, safety_radius), 0.0, 1.0);
        return 18.0 + 10.0 * near_penalty + 0.5 * static_cast<double>(hit_count);
    };
    const auto lookup_grid_cost = [&](const Eigen::Vector3d& p_b) {
        if (p_b.x() <= 0.0) return 0.0;
        const double bearing = std::atan2(p_b.y(), std::max(1e-3, p_b.x()));
        const double elev = std::atan2(p_b.z(), std::max(1e-3, p_b.head<2>().norm()));
        if (bearing < grid.yaw_min || bearing > grid.yaw_max) return 0.0;
        if (elev < grid.pitch_min || elev > grid.pitch_max) return 0.0;
        const double yaw_ratio = (bearing - grid.yaw_min) / std::max(1e-3, grid.yaw_max - grid.yaw_min);
        const double pitch_ratio = (elev - grid.pitch_min) / std::max(1e-3, grid.pitch_max - grid.pitch_min);
        const int yaw_bin = std::max(0, std::min(grid.yaw_bins - 1,
            static_cast<int>(std::floor(yaw_ratio * static_cast<double>(grid.yaw_bins)))));
        const int pitch_bin = std::max(0, std::min(grid.pitch_bins - 1,
            static_cast<int>(std::floor(pitch_ratio * static_cast<double>(grid.pitch_bins)))));
        const size_t flat = static_cast<size_t>(pitch_bin * grid.yaw_bins + yaw_bin);
        const double range_norm =
            (grid.point_counts[flat] > 0)
                ? (1.0 - clampd(grid.min_ranges[flat] / std::max(1e-3, grid.max_range), 0.0, 1.0))
                : 0.0;
        return 2.0 * clampd(grid.cell_costs[flat], 0.0, 1.0) + 0.8 * range_norm;
    };
    const auto trajectory_field_cost = [&](const Eigen::Vector3d& seg_start_b,
                                           const Eigen::Vector3d& seg_mid_b,
                                           const Eigen::Vector3d& seg_end_b) {
        // Sample the candidate path directly against the direction-space field.
        const double sample_spacing = std::max(0.4, stage2_local_traj_spacing_);
        double cost = 0.0;
        auto accumulate_segment = [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b, double weight) {
            const Eigen::Vector3d seg = b - a;
            const double seg_len = seg.norm();
            if (seg_len < 1e-3) return;
            const int steps = std::max(1, static_cast<int>(std::ceil(seg_len / sample_spacing)));
            for (int s = 1; s <= steps; ++s) {
                const double alpha = static_cast<double>(s) / static_cast<double>(steps);
                const Eigen::Vector3d sample_b = a + alpha * seg;
                cost += weight * lookup_grid_cost(sample_b);
            }
        };
        accumulate_segment(seg_start_b, seg_mid_b, 1.0);
        accumulate_segment(seg_mid_b, seg_end_b, 0.7);
        return cost;
    };
    for (int pidx = 0; pidx < grid.pitch_bins; ++pidx) {
        const double pitch_center = grid.pitch_min +
            (static_cast<double>(pidx) + 0.5) * pitch_step;
        if (std::find(allowed_pitch_bins.begin(), allowed_pitch_bins.end(), pidx) == allowed_pitch_bins.end()) {
            for (int yidx = 0; yidx < grid.yaw_bins; ++yidx) {
                distance_cost(pidx, yidx) = 1e4f;
                other_cost(pidx, yidx) = 1e4f;
            }
            continue;
        }
        for (int yidx = 0; yidx < grid.yaw_bins; ++yidx) {
            const double yaw_center = grid.yaw_min +
                (static_cast<double>(yidx) + 0.5) * yaw_step;
            if (yaw_center < goal_yaw - max_deflect || yaw_center > goal_yaw + max_deflect) {
                distance_cost(pidx, yidx) = 1e4f;
                other_cost(pidx, yidx) = 1e4f;
                continue;
            }
            if (std::fabs(pitch_center) > candidate_pitch_limit) {
                distance_cost(pidx, yidx) = 1e4f;
                other_cost(pidx, yidx) = 1e4f;
                continue;
            }
            const size_t flat = static_cast<size_t>(pidx * grid.yaw_bins + yidx);
            const double obstacle_distance =
                (grid.point_counts[flat] > 0)
                    ? std::min(grid.mean_ranges[flat], grid.min_ranges[flat])
                    : 0.0;
            if (obstacle_distance > 0.0) {
                const double d = distance_bias - obstacle_distance;
                distance_cost(pidx, yidx) =
                    static_cast<float>(stage2_avoid_clearance_weight_ *
                                       100.0 * (1.0 + d / std::sqrt(1.0 + d * d)));
            }
            const double horizontal_len = probe_len * std::cos(pitch_center);
            const Eigen::Vector3d candidate_target_b(horizontal_len * std::cos(yaw_center),
                                                     horizontal_len * std::sin(yaw_center),
                                                     probe_len * std::sin(pitch_center));
            const Eigen::Vector3d candidate_target_w = current_pos + R_wb * candidate_target_b;
            const Eigen::Vector3d candidate_delta_w = candidate_target_w - raw_target;
            const double centerline_offset = candidate_delta_w.dot(path_lateral_w);
            const double centerline_vertical_offset = candidate_delta_w.z();
            const double yaw_cost = stage2_avoid_alignment_weight_ *
                                    stage2_angle_diff(yaw_center, goal_yaw) *
                                    stage2_angle_diff(yaw_center, goal_yaw);
            const double continuity_cost =
                have_last_avoid_yaw ? 0.5 * stage2_angle_diff(yaw_center, last_avoid_yaw) *
                                          stage2_angle_diff(yaw_center, last_avoid_yaw)
                                    : 0.0;
            const double line_cost = stage2_avoid_offset_weight_ *
                                     std::pow(centerline_offset / std::max(0.5, corridor_half_width_safe), 2.0);
            const double pitch_cost =
                stage2_avoid_vertical_weight_ * 4.0 * pitch_center * pitch_center +
                stage2_avoid_vertical_rejoin_weight_ *
                    std::pow(centerline_vertical_offset / std::max(0.3, corridor_half_height_safe), 2.0);
            const double occupancy_cost =
                stage2_avoid_density_weight_ * clampd(grid.cell_costs[flat], 0.0, 1.0);
            const double gap_width = estimate_gap_width(pidx, yidx,
                                                        (grid.point_counts[flat] > 0) ? grid.min_ranges[flat] : probe_len);
            const double gap_shortfall =
                clampd((stage2_gap_safe_width_ - gap_width) / std::max(0.1, stage2_gap_safe_width_), 0.0, 1.0);
            const double gap_cost =
                stage2_gap_width_weight_ * 8.0 * gap_shortfall * gap_shortfall;
            double corridor_cost = 0.0;
            const double abs_centerline_offset = std::fabs(centerline_offset);
            if (abs_centerline_offset > soft_corridor_half_width) {
                const double corridor_ratio =
                    clampd((abs_centerline_offset - soft_corridor_half_width) /
                               std::max(1e-3, corridor_half_width_safe - soft_corridor_half_width),
                           0.0, 1.0);
                corridor_cost = 8.0 * corridor_ratio * corridor_ratio;
            }
            if (abs_centerline_offset >= corridor_half_width_safe) {
                corridor_cost += 50.0;
            }
            const double abs_vertical_offset = std::fabs(centerline_vertical_offset);
            if (abs_vertical_offset > soft_corridor_half_height) {
                const double corridor_ratio_z =
                    clampd((abs_vertical_offset - soft_corridor_half_height) /
                               std::max(1e-3, corridor_half_height_safe - soft_corridor_half_height),
                           0.0, 1.0);
                corridor_cost += 8.0 * corridor_ratio_z * corridor_ratio_z;
            }
            if (abs_vertical_offset >= corridor_half_height_safe) {
                corridor_cost += 50.0;
            }
            const double path_collision_cost =
                segment_collision_cost(Eigen::Vector3d::Zero(), candidate_target_b) +
                0.6 * segment_collision_cost(candidate_target_b, raw_target_b);
            const double traj_cost =
                trajectory_field_cost(Eigen::Vector3d::Zero(), candidate_target_b, raw_target_b);
            other_cost(pidx, yidx) =
                static_cast<float>(yaw_cost + continuity_cost + line_cost + pitch_cost + occupancy_cost + gap_cost + corridor_cost + path_collision_cost + traj_cost);
        }
    }

    Eigen::MatrixXf distance_cost_smoothed = distance_cost;
    smooth_stage2_polar_matrix(distance_cost_smoothed, 1, std::max(1, stage2_process_scale_factor_ / 2));
    const Eigen::MatrixXf total_cost = other_cost + distance_cost_smoothed;

    std::vector<Stage2CandidateCell> candidates;
    candidates.reserve(static_cast<size_t>(grid.pitch_bins * grid.yaw_bins));
    int dbg_window_cells = 0;  // [METRIC] 落在目标扇区+pitch限内的候选窗口总数
    for (int pidx = 0; pidx < grid.pitch_bins; ++pidx) {
        const double pitch_center = grid.pitch_min +
            (static_cast<double>(pidx) + 0.5) * pitch_step;
        for (int yidx = 0; yidx < grid.yaw_bins; ++yidx) {
            const double yaw_center = grid.yaw_min +
                (static_cast<double>(yidx) + 0.5) * yaw_step;
            if (yaw_center < goal_yaw - max_deflect || yaw_center > goal_yaw + max_deflect) continue;
            if (std::fabs(pitch_center) > candidate_pitch_limit) continue;
            ++dbg_window_cells;
            const float cost = total_cost(pidx, yidx);
            if (!std::isfinite(cost) || cost >= 1e4f) continue;
            candidates.push_back(Stage2CandidateCell{static_cast<double>(cost), pidx, yidx});
        }
    }
    // [METRIC] 记录候选可行率（供 STAGE2_PATH 日志输出）
    stage2_debug_feasible_candidates_ = static_cast<int>(candidates.size());
    stage2_debug_total_candidates_ = dbg_window_cells;
    stage2_debug_fallback_used_ = false;
    std::sort(candidates.begin(), candidates.end(),
              [](const Stage2CandidateCell& a, const Stage2CandidateCell& b) { return a.cost < b.cost; });

    // Prefer staying on the previous candidate unless the new one is clearly better.
    if (!candidates.empty() &&
        stage2_last_candidate_yaw_idx_ >= 0 &&
        stage2_last_candidate_pitch_idx_ >= 0) {
        for (const auto& candidate : candidates) {
            if (candidate.yaw_idx == stage2_last_candidate_yaw_idx_ &&
                candidate.pitch_idx == stage2_last_candidate_pitch_idx_) {
                const double best_cost = candidates.front().cost;
                if (candidate.cost <= best_cost + stage2_avoid_candidate_switch_margin_) {
                    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                        [&](const Stage2CandidateCell& c) {
                            return c.yaw_idx == candidate.yaw_idx && c.pitch_idx == candidate.pitch_idx;
                        }), candidates.end());
                    candidates.insert(candidates.begin(), candidate);
                }
                break;
            }
        }
    }

    const size_t debug_count = std::min(candidates.size(), static_cast<size_t>(candidate_count));
    for (size_t i = 0; i < debug_count; ++i) {
        const int pidx = candidates[i].pitch_idx;
        const int yidx = candidates[i].yaw_idx;
        const double yaw_center = grid.yaw_min +
            (static_cast<double>(yidx) + 0.5) * yaw_step;
        const double pitch_center = grid.pitch_min +
            (static_cast<double>(pidx) + 0.5) * pitch_step;
        const double horizontal_len = probe_len * std::cos(pitch_center);
        const Eigen::Vector3d candidate_target =
            current_pos + R_wb * Eigen::Vector3d(horizontal_len * std::cos(yaw_center),
                                                 horizontal_len * std::sin(yaw_center),
                                                 probe_len * std::sin(pitch_center));
        const size_t flat = static_cast<size_t>(pidx * grid.yaw_bins + yidx);
        stage2_debug_candidate_targets_.push_back(Eigen::Vector3d(candidate_target.x(),
                                                                  candidate_target.y(),
                                                                  candidate_target.z()));
        stage2_debug_candidate_scores_.push_back(-candidates[i].cost);
        stage2_debug_candidate_clearances_.push_back(
            (grid.point_counts[flat] > 0) ? grid.min_ranges[flat] : grid.max_range);
        stage2_debug_candidate_gap_widths_.push_back(
            estimate_gap_width(pidx, yidx,
                               (grid.point_counts[flat] > 0) ? grid.min_ranges[flat] : probe_len));
    }

    if (candidates.empty()) {
        // Fallback: choose the less occupied lateral side instead of pushing straight ahead.
        double left_occ = 0.0;
        double right_occ = 0.0;
        const int center_yaw_idx = std::max(0, std::min(grid.yaw_bins - 1,
            static_cast<int>(std::floor((goal_yaw - grid.yaw_min) /
                                        std::max(1e-3, grid.yaw_max - grid.yaw_min) *
                                        static_cast<double>(grid.yaw_bins)))));
        for (int pidx = 0; pidx < grid.pitch_bins; ++pidx) {
            for (int y = 0; y < grid.yaw_bins; ++y) {
                const size_t flat = static_cast<size_t>(pidx * grid.yaw_bins + y);
                const double occ = static_cast<double>(grid.point_counts[flat]) + 2.0 * grid.cell_costs[flat];
                if (y < center_yaw_idx) {
                    left_occ += occ;
                } else if (y > center_yaw_idx) {
                    right_occ += occ;
                }
            }
        }
        const double emergency_yaw = (left_occ <= right_occ) ? -max_deflect : max_deflect;
        const Eigen::Vector3d emergency_target_b(probe_len * std::cos(emergency_yaw),
                                                 probe_len * std::sin(emergency_yaw),
                                                 0.0);
        avoid_offset = clampd(emergency_target_b.y() - raw_target_b.y(), -max_offset, max_offset);
        avoid_target = current_pos + R_wb * Eigen::Vector3d(emergency_target_b.x(),
                                                            raw_target_b.y() + avoid_offset,
                                                            raw_target_b.z());
        stage2_debug_selected_candidate_ = -2;
        stage2_debug_fallback_used_ = true;
        stage2_debug_best_candidate_clearance_ = std::min(stage2_debug_best_candidate_clearance_, nearest_center_range);
        stage2_debug_best_gap_width_ = 0.0;
        return true;
    }

    const Stage2CandidateCell& best_cell = candidates.front();
    const double best_yaw = grid.yaw_min +
        (static_cast<double>(best_cell.yaw_idx) + 0.5) * yaw_step;
    const double best_pitch = grid.pitch_min +
        (static_cast<double>(best_cell.pitch_idx) + 0.5) * pitch_step;
    const size_t best_flat = static_cast<size_t>(best_cell.pitch_idx * grid.yaw_bins + best_cell.yaw_idx);
    const double best_horizontal_len = probe_len * std::cos(best_pitch);
    const Eigen::Vector3d best_target_b(best_horizontal_len * std::cos(best_yaw),
                                        best_horizontal_len * std::sin(best_yaw),
                                        probe_len * std::sin(best_pitch));
    stage2_debug_selected_candidate_ = 0;
    stage2_debug_best_candidate_clearance_ =
        (grid.point_counts[best_flat] > 0) ? grid.min_ranges[best_flat] : grid.max_range;
    stage2_debug_best_gap_width_ =
        estimate_gap_width(best_cell.pitch_idx, best_cell.yaw_idx,
                           (grid.point_counts[best_flat] > 0) ? grid.min_ranges[best_flat] : probe_len);
    stage2_last_candidate_yaw_idx_ = best_cell.yaw_idx;
    stage2_last_candidate_pitch_idx_ = best_cell.pitch_idx;
    avoid_offset = clampd((best_target_b.y() - raw_target_b.y()) * activation,
                          -max_offset,
                          max_offset);
    const double target_z_b = raw_target_b.z() + activation * (best_target_b.z() - raw_target_b.z());
    avoid_target = current_pos + R_wb * Eigen::Vector3d(best_target_b.x(),
                                                        raw_target_b.y() + avoid_offset,
                                                        target_z_b);
    return true;
}

// -----------------------------------------------------------------------------
// Stage2 visualization helpers
// -----------------------------------------------------------------------------
void BasicDev::publish_stage2_obstacle_points() const
{
    if (!stage2_active_ || !stage2_avoid_enable_ || !stage2_frame_cloud_body_ || stage2_frame_cloud_body_->points.empty()) {
        return;
    }

    pcl::PointCloud<pcl::PointXYZ> selected;
    selected.points.reserve(stage2_frame_cloud_body_->points.size());

    const Eigen::Matrix3d R_wb = current_quat.normalized().toRotationMatrix();
    const double x_max = std::max(0.5, stage2_avoid_forward_range_);
    const double corridor_half_height_safe =
        std::max(0.3,
                 stage2_corridor_half_height_safe_);
    const double z_lim = std::max(stage2_avoid_vertical_half_height_, corridor_half_height_safe);

    for (const auto& p : stage2_frame_cloud_body_->points) {
        const Eigen::Vector3d p_b(p.x, p.y, p.z);
        if (p_b.x() <= 0.0) continue;
        if (std::fabs(p_b.z()) > z_lim) continue;
        if (p_b.head<2>().norm() > x_max) continue;
        const Eigen::Vector3d p_w = current_pos + R_wb * p_b;
        selected.points.emplace_back(static_cast<float>(p_w.x()),
                                     static_cast<float>(p_w.y()),
                                     static_cast<float>(p_w.z()));
    }

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(selected, msg);
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = viz_frame_id_;
    stage2_obstacle_points_pub_.publish(msg);
}

void BasicDev::publish_stage2_debug_markers(const Eigen::Vector3d& raw_target,
                                            const Eigen::Vector3d& avoid_target,
                                            double obs_dist,
                                            double avoid_offset,
                                            double avoid_scale) const
{
    if (!stage2_active_ || !stage2_avoid_enable_) {
        return;
    }

    auto make_marker = [&](int id, int type) {
        visualization_msgs::Marker m;
        m.header.stamp = ros::Time::now();
        m.header.frame_id = viz_frame_id_;
        m.ns = "stage2_avoid";
        m.id = id;
        m.type = type;
        m.action = visualization_msgs::Marker::ADD;
        m.pose.orientation.w = 1.0;
        m.color.a = 1.0;
        return m;
    };

    visualization_msgs::Marker raw = make_marker(0, visualization_msgs::Marker::SPHERE);
    raw.pose.position.x = raw_target.x();
    raw.pose.position.y = raw_target.y();
    raw.pose.position.z = raw_target.z();
    raw.scale.x = raw.scale.y = raw.scale.z = 0.30;
    raw.color.r = 1.0;
    raw.color.g = 0.85;
    raw.color.b = 0.10;
    stage2_debug_marker_pub_.publish(raw);

    visualization_msgs::Marker avoid = make_marker(1, visualization_msgs::Marker::SPHERE);
    avoid.pose.position.x = avoid_target.x();
    avoid.pose.position.y = avoid_target.y();
    avoid.pose.position.z = avoid_target.z();
    avoid.scale.x = avoid.scale.y = avoid.scale.z = 0.32;
    avoid.color.r = 0.10;
    avoid.color.g = 0.95;
    avoid.color.b = 0.15;
    stage2_debug_marker_pub_.publish(avoid);

    visualization_msgs::Marker box = make_marker(2, visualization_msgs::Marker::CUBE);
    const double box_x = std::max(0.5, stage2_avoid_forward_range_);
    box.pose.position.x = current_pos.x();
    box.pose.position.y = current_pos.y();
    box.pose.position.z = current_pos.z();
    const Eigen::Quaterniond q_box = current_quat.normalized();
    box.pose.orientation.x = q_box.x();
    box.pose.orientation.y = q_box.y();
    box.pose.orientation.z = q_box.z();
    box.pose.orientation.w = q_box.w();
    box.scale.x = box_x;
    box.scale.y = std::max(0.6, 2.0 * stage2_avoid_side_probe_width_);
    box.scale.z = std::max(0.6, 2.0 * stage2_avoid_vertical_half_height_);
    // Move box center forward by half length along body x.
    const Eigen::Vector3d box_center = current_pos + q_box.toRotationMatrix() * Eigen::Vector3d(box_x * 0.5, 0.0, 0.0);
    box.pose.position.x = box_center.x();
    box.pose.position.y = box_center.y();
    box.pose.position.z = box_center.z();
    box.color.a = 0.12;
    box.color.r = 0.2;
    box.color.g = 0.6;
    box.color.b = 1.0;
    stage2_debug_marker_pub_.publish(box);

    visualization_msgs::Marker raw_arrow = make_marker(3, visualization_msgs::Marker::ARROW);
    geometry_msgs::Point p0, p1;
    p0.x = current_pos.x();
    p0.y = current_pos.y();
    p0.z = current_pos.z();
    p1.x = raw_target.x();
    p1.y = raw_target.y();
    p1.z = raw_target.z();
    raw_arrow.points.push_back(p0);
    raw_arrow.points.push_back(p1);
    raw_arrow.scale.x = 0.08;
    raw_arrow.scale.y = 0.16;
    raw_arrow.scale.z = 0.16;
    raw_arrow.color.r = 1.0;
    raw_arrow.color.g = 0.75;
    raw_arrow.color.b = 0.10;
    stage2_debug_marker_pub_.publish(raw_arrow);

    visualization_msgs::Marker avoid_arrow = make_marker(4, visualization_msgs::Marker::ARROW);
    geometry_msgs::Point p2;
    p2.x = avoid_target.x();
    p2.y = avoid_target.y();
    p2.z = avoid_target.z();
    avoid_arrow.points.push_back(p0);
    avoid_arrow.points.push_back(p2);
    avoid_arrow.scale.x = 0.10;
    avoid_arrow.scale.y = 0.18;
    avoid_arrow.scale.z = 0.18;
    avoid_arrow.color.r = 0.10;
    avoid_arrow.color.g = 0.95;
    avoid_arrow.color.b = 0.15;
    stage2_debug_marker_pub_.publish(avoid_arrow);

    visualization_msgs::Marker text = make_marker(5, visualization_msgs::Marker::TEXT_VIEW_FACING);
    text.pose.position.x = current_pos.x();
    text.pose.position.y = current_pos.y();
    text.pose.position.z = current_pos.z() + 1.5;
    text.scale.z = 0.45;
    text.color.r = 1.0;
    text.color.g = 1.0;
    text.color.b = 1.0;
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << "obs=";
    if (std::isfinite(obs_dist)) {
        oss << obs_dist;
    } else {
        oss << "inf";
    }
    oss << "  pts=" << stage2_debug_blockage_points_
        << "  dens=" << stage2_debug_blockage_density_
        << "  bins=" << stage2_debug_blockage_occupied_bins_
        << "  best=";
    if (std::isfinite(stage2_debug_best_candidate_clearance_)) {
        oss << stage2_debug_best_candidate_clearance_;
    } else {
        oss << "inf";
    }
    oss << "  gapw=" << stage2_debug_best_gap_width_;
    oss << "  off=" << avoid_offset << "  scale=" << avoid_scale;
    if (stage2_debug_selected_candidate_ >= 0 &&
        stage2_debug_selected_candidate_ < static_cast<int>(stage2_debug_candidate_clearances_.size())) {
        oss << "  cand=" << stage2_debug_selected_candidate_
            << "  clr=" << stage2_debug_candidate_clearances_[static_cast<size_t>(stage2_debug_selected_candidate_)];
        if (stage2_debug_selected_candidate_ < static_cast<int>(stage2_debug_candidate_gap_widths_.size())) {
            oss << "  gw=" << stage2_debug_candidate_gap_widths_[static_cast<size_t>(stage2_debug_selected_candidate_)];
        }
    }
    text.text = oss.str();
    stage2_debug_marker_pub_.publish(text);

    if (!stage2_debug_candidate_targets_.empty()) {
        visualization_msgs::Marker rays = make_marker(6, visualization_msgs::Marker::LINE_LIST);
        rays.scale.x = 0.035;
        for (size_t i = 0; i < stage2_debug_candidate_targets_.size(); ++i) {
            geometry_msgs::Point a = p0;
            geometry_msgs::Point b;
            b.x = stage2_debug_candidate_targets_[i].x();
            b.y = stage2_debug_candidate_targets_[i].y();
            b.z = stage2_debug_candidate_targets_[i].z();
            rays.points.push_back(a);
            rays.points.push_back(b);

            std_msgs::ColorRGBA c;
            c.a = 0.85f;
            if (static_cast<int>(i) == stage2_debug_selected_candidate_) {
                c.r = 0.10f; c.g = 0.95f; c.b = 0.15f;
            } else {
                c.r = 0.45f; c.g = 0.45f; c.b = 0.95f;
            }
            rays.colors.push_back(c);
            rays.colors.push_back(c);
        }
        stage2_debug_marker_pub_.publish(rays);

        visualization_msgs::Marker endpoints = make_marker(7, visualization_msgs::Marker::SPHERE_LIST);
        endpoints.scale.x = 0.16;
        endpoints.scale.y = 0.16;
        endpoints.scale.z = 0.16;
        for (size_t i = 0; i < stage2_debug_candidate_targets_.size(); ++i) {
            geometry_msgs::Point pt;
            pt.x = stage2_debug_candidate_targets_[i].x();
            pt.y = stage2_debug_candidate_targets_[i].y();
            pt.z = stage2_debug_candidate_targets_[i].z();
            endpoints.points.push_back(pt);

            const double score = (i < stage2_debug_candidate_scores_.size()) ? stage2_debug_candidate_scores_[i] : 0.0;
            const double norm = clampd(0.5 + 0.25 * score, 0.0, 1.0);
            std_msgs::ColorRGBA c;
            c.a = 1.0f;
            if (static_cast<int>(i) == stage2_debug_selected_candidate_) {
                c.r = 0.10f; c.g = 1.0f; c.b = 0.15f;
            } else {
                c.r = static_cast<float>(1.0 - norm);
                c.g = static_cast<float>(norm);
                c.b = 0.20f;
            }
            endpoints.colors.push_back(c);
        }
        stage2_debug_marker_pub_.publish(endpoints);
    }
}

// -----------------------------------------------------------------------------
// Stage1 controller: tuned CSV path follower
// -----------------------------------------------------------------------------
void BasicDev::path_follow_step(double& vx, double& vy, double& vz, double& yaw_deg, bool& finished)
{
    // 单步路径跟踪：
    // 1) 找最近点 + 前视点
    // 2) 计算世界系误差并变换到机体系
    // 3) P控制输出速度 + 限幅
    // 4) 大角误差时先转向后平移
    finished = false;
    vx = vy = vz = yaw_deg = 0.0;
    if (!path_loaded_ || path_pts_.size() < 2) return;

    const size_t i_near = find_nearest_index(current_pos);
    const double s_now = (i_near < path_s_.size()) ? path_s_[i_near] : 0.0;
    const ros::Time xy_now = ros::Time::now();
    double horiz_speed = 0.0;
    if (!xy_speed_inited_) {
        xy_speed_inited_ = true;
        prev_pos_for_xy_vel_ = current_pos;
        prev_xy_vel_time_ = xy_now;
    } else {
        const double dt_xy = clampd((xy_now - prev_xy_vel_time_).toSec(), 0.005, 0.10);
        const Eigen::Vector3d dp = current_pos - prev_pos_for_xy_vel_;
        horiz_speed = std::hypot(dp.x(), dp.y()) / std::max(1e-3, dt_xy);
        prev_pos_for_xy_vel_ = current_pos;
        prev_xy_vel_time_ = xy_now;
    }
    double lookahead_now = clampd(lookahead_dist_ +
                                  lookahead_time_gain_ * std::max(0.0, horiz_speed - lookahead_speed_ref_),
                                  lookahead_min_dist_,
                                  lookahead_max_dist_);
    const bool stage1_turn_guard_active =
        (!stage2_active_) &&
        stage1_turn_guard_enable_ &&
        (s_now >= stage1_turn_guard_s_min_) &&
        (s_now <= stage1_turn_guard_s_max_);
    if (stage1_turn_guard_active) {
        lookahead_now = std::max(lookahead_now, stage1_turn_guard_lookahead_);
    }
    const size_t i_tar  = find_lookahead_index(i_near, lookahead_now);
    const Eigen::Vector3d p_tar_raw = path_pts_[i_tar];
    const Eigen::Vector3d p_tar = p_tar_raw;
    const Eigen::Matrix3d R_wb = current_quat.normalized().toRotationMatrix(); // body->world
    publish_target_marker(p_tar);

    const Eigen::Vector3d e_w = p_tar - current_pos;

    // 用当前姿态把误差从世界系转换到机体系，避免直接用欧拉角带来的符号混淆
    const Eigen::Vector3d e_b  = R_wb.transpose() * e_w;                        // world->body

    // yaw误差（命令符号约定）：正值=顺时针需求，使用与XY一致的前视点
    const double yaw_err = std::atan2(e_b.y(), e_b.x());
    const double yaw_err_deg = yaw_err * 180.0 / M_PI;
    // 偏航角速度估计（由姿态差分得到，CCW为正，单位deg/s）
    const ros::Time yaw_now = ros::Time::now();
    double yaw_rate_meas_ccw = 0.0;
    const double yaw_cur_rad = quat_to_yaw_rad(current_quat);
    if (!yaw_rate_inited_) {
        yaw_rate_inited_ = true;
        yaw_prev_rad_ = yaw_cur_rad;
        yaw_prev_time_ = yaw_now;
        yaw_rate_meas_filt_ = 0.0;
    } else {
        const double dt_yaw = clampd((yaw_now - yaw_prev_time_).toSec(), 0.005, 0.10);
        const double yaw_delta = wrap_pi_rad(yaw_cur_rad - yaw_prev_rad_);
        const double yaw_rate_raw = yaw_delta * 180.0 / M_PI / std::max(1e-3, dt_yaw);
        yaw_rate_meas_filt_ =
            (1.0 - yaw_rate_lpf_alpha_) * yaw_rate_meas_filt_ + yaw_rate_lpf_alpha_ * yaw_rate_raw;
        yaw_rate_meas_ccw = yaw_rate_meas_filt_;
        yaw_prev_rad_ = yaw_cur_rad;
        yaw_prev_time_ = yaw_now;
    }

    // yaw PD控制（输出约定：正值=顺时针）
    // yaw_err_deg 正表示应顺时针；yaw_rate_meas_ccw 正表示当前逆时针
    // 因此 D项使用 +Kd*omega_ccw：当机体逆时针转动时增大顺时针指令，抑制反向摆动
    if (std::fabs(yaw_err_deg) <= yaw_stop_tolerance_deg_) {
        yaw_deg = 0.0;
    } else {
        const double yaw_kd_now =
            (stage2_mode_ == Stage2FollowMode::AVOID) ? 0.5 * fp_kd_yaw_rate_ : fp_kd_yaw_rate_;
        const double yaw_limit_now =
            (stage2_mode_ == Stage2FollowMode::AVOID) ? std::min(yaw_soft_limit_deg_, 35.0) : yaw_soft_limit_deg_;
        yaw_deg = fp_kp_yaw_ * yaw_err_deg + yaw_kd_now * yaw_rate_meas_ccw;
        yaw_deg = clampd(yaw_deg, -yaw_limit_now, yaw_limit_now);
        if (std::fabs(yaw_deg) < yaw_min_cmd_deg_) {
            yaw_deg = (yaw_err_deg >= 0.0) ? yaw_min_cmd_deg_ : -yaw_min_cmd_deg_;
        }
    }

    // 平面速度控制：误差越大速度越大，然后做上限限幅
    double vx_limit_now = stage1_turn_guard_active ? std::min(max_vx_, stage1_turn_guard_max_vx_) : max_vx_;
    const double vy_limit_now = stage1_turn_guard_active ? std::min(max_vy_, stage1_turn_guard_max_vy_) : max_vy_;
    vx = clampd(fp_kp_x_ * e_b.x(), -vx_limit_now, vx_limit_now);
    vy = clampd(fp_kp_y_ * e_b.y(), -vy_limit_now, vy_limit_now);
    // 小改动：只在横向偏差超过阈值时才降低前向推进，避免全程被拖慢
    double vx_lateral_scale = 1.0;
    const double ey_abs = std::fabs(e_b.y());
    double lateral_thresh_now = lateral_slow_err_thresh_;
    const ros::Time t_now_for_gate = ros::Time::now();
    if (!takeoff_success_time.isZero()) {
        const double flight_time = (t_now_for_gate - takeoff_success_time).toSec();
        if (flight_time >= gate_relax_time_sec_) {
            lateral_thresh_now = late_lateral_err_thresh_;
        }
    }
    if (ey_abs > lateral_thresh_now) {
        const double ey_excess = ey_abs - lateral_thresh_now;
        vx_lateral_scale = clampd(1.0 / (1.0 + lateral_slow_gain_ * ey_excess), lateral_slow_min_scale_, 1.0);
    }
    vx *= vx_lateral_scale;
    // 抑制“贴线段”左右摇晃：仅在横向误差较小时对vy做轻量低通，不影响大偏差快速纠偏
    const double vy_alpha = clampd(fp_vy_lpf_alpha_, 0.0, 1.0);
    if (!xy_vel_inited_) {
        vy_cmd_filt_ = vy;
        xy_vel_inited_ = true;
    } else if (ey_abs < 4.0) {
        vy_cmd_filt_ = (1.0 - vy_alpha) * vy_cmd_filt_ + vy_alpha * vy;
        vy = vy_cmd_filt_;
    } else {
        vy_cmd_filt_ = vy;
    }
    // 高度控制：传统单环PID（误差->PID->vz）
    const size_t i_tar_z = find_lookahead_index(i_near, std::max(0.0, z_lookahead_dist_));
    const double z_target = path_pts_[i_tar_z].z();
    const double dz_w = z_target - current_pos.z(); // 内部采用z-up：目标更高时dz_w>0

    // --- Z轴PID核心（D项使用速度环：v_err = v_ref - v_meas） ---
    const ros::Time z_now = ros::Time::now();
    double dt = 0.03;
    if (z_pid_inited_) {
        dt = clampd((z_now - z_pid_prev_time_).toSec(), 0.005, 0.10);
    } else {
        z_pid_int_ = 0.0;
        z_pid_prev_err_ = 0.0;
        z_pid_d_filt_ = 0.0;
        z_pid_inited_ = true;
    }

    // 初始化速度环状态
    if (!z_vel_inited_) {
        z_prev_meas_ = current_pos.z();
        z_prev_ref_ = z_target;
        z_vel_meas_filt_ = 0.0;
        z_ref_vel_filt_ = 0.0;
        z_vel_inited_ = true;
    }

    // 估计实际竖直速度（来自当前位置差分）
    const double v_meas_raw = (current_pos.z() - z_prev_meas_) / std::max(1e-3, dt);
    z_vel_meas_filt_ =
        (1.0 - z_vel_lpf_alpha_) * z_vel_meas_filt_ + z_vel_lpf_alpha_ * v_meas_raw;
    z_prev_meas_ = current_pos.z();

    // 估计目标竖直速度（来自目标z变化差分），用于速度前馈
    const double v_ref_raw = (z_target - z_prev_ref_) / std::max(1e-3, dt);
    z_ref_vel_filt_ =
        (1.0 - z_ref_vel_lpf_alpha_) * z_ref_vel_filt_ + z_ref_vel_lpf_alpha_ * v_ref_raw;
    z_prev_ref_ = z_target;

    const double v_ref = clampd(z_ref_vel_ff_gain_ * z_ref_vel_filt_, -max_vz_down_, max_vz_);
    const double v_err = v_ref - z_vel_meas_filt_;

    // 误差符号切换时衰减积分，减少“反向拖尾”
    if ((dz_w > 0.0 && z_pid_int_ < 0.0) || (dz_w < 0.0 && z_pid_int_ > 0.0)) {
        z_pid_int_ *= 0.5;
    }

    z_pid_d_filt_ = (1.0 - z_pid_d_lpf_alpha_) * z_pid_d_filt_ + z_pid_d_lpf_alpha_ * v_err;

    // 先用候选积分计算一次，做抗饱和判定
    const double int_candidate = clampd(z_pid_int_ + dz_w * dt, -z_pid_i_limit_, z_pid_i_limit_);
    const double u_unsat_candidate = z_pid_kp_ * dz_w + z_pid_ki_ * int_candidate + z_pid_kd_ * z_pid_d_filt_;
    const double u_sat_candidate = clampd(u_unsat_candidate, -max_vz_down_, max_vz_);
    const bool allow_integrate =
        (std::fabs(u_unsat_candidate - u_sat_candidate) < 1e-6) ||
        ((u_unsat_candidate > u_sat_candidate) && (dz_w < 0.0)) ||
        ((u_unsat_candidate < u_sat_candidate) && (dz_w > 0.0));
    if (allow_integrate) {
        z_pid_int_ = int_candidate;
    }

    const double u_unsat = z_pid_kp_ * dz_w + z_pid_ki_ * z_pid_int_ + z_pid_kd_ * z_pid_d_filt_;
    vz = clampd(u_unsat, -max_vz_down_, max_vz_);

    z_pid_prev_err_ = dz_w;
    z_pid_prev_time_ = z_now;

    // 角误差过大时做“降速门控”（不再清零），避免横冲导致偏航/撞墙
    if (enable_yaw_only_gate_ && std::fabs(yaw_err_deg) > yaw_only_deg_) {
        // 航向还没对齐时同时压低前向和横向推进，减少“边转边横移”导致的摆动
        const double kMinScaleVx = 0.20;
        const double kMinScaleVy = 0.55;
        const double scale_vx = clampd(yaw_only_deg_ / std::fabs(yaw_err_deg), kMinScaleVx, 1.0);
        const double scale_vy = clampd(std::sqrt(scale_vx), kMinScaleVy, 1.0);
        vx *= scale_vx;
        vy *= scale_vy;
    }

    const double dist_end = (path_pts_.back() - current_pos).norm();
    if (dist_end < final_brake_start_dist_ && final_brake_start_dist_ > final_stop_before_end_dist_ + 1e-3) {
        const double brake_ratio = clampd((dist_end - final_stop_before_end_dist_) /
                                          (final_brake_start_dist_ - final_stop_before_end_dist_),
                                          final_brake_min_scale_, 1.0);
        vx *= brake_ratio;
        vy *= brake_ratio;
        vz *= brake_ratio;
    }

    if (dist_end < final_stop_before_end_dist_) finished = true;

    ROS_INFO_THROTTLE(0.5, "[PATH] near=%zu tar=%zu tar_z=%zu s=%.1f guard=%d L=%.2f vh=%.2f yaw_err=%.1f wccw=%.1f dz=%.2f vref=%.2f vz=%.2f zi=%.2f vd=%.2f dist_end=%.2f e_b=(%.2f %.2f %.2f) cmd=(%.2f %.2f %.2f %.1f)",
                      i_near, i_tar, i_tar_z, s_now, static_cast<int>(stage1_turn_guard_active), lookahead_now, horiz_speed, yaw_err_deg, yaw_rate_meas_ccw, dz_w,
                      v_ref, z_vel_meas_filt_, z_pid_int_, z_pid_d_filt_, dist_end, e_b.x(), e_b.y(), e_b.z(), vx, vy, vz, yaw_deg);
}

// -----------------------------------------------------------------------------
// Stage2 controller: continuous replanning + local trajectory tracking
// -----------------------------------------------------------------------------
void BasicDev::stage2_follow_step(double& vx, double& vy, double& vz, double& yaw_deg, bool& finished)
{
    finished = false;
    vx = vy = vz = yaw_deg = 0.0;
    stage2_mode_ = Stage2FollowMode::CRUISE;
    if (!path_loaded_ || path_pts_.size() < 2) return;

    const size_t i_near = find_nearest_index(current_pos);
    const ros::Time xy_now = ros::Time::now();
    double horiz_speed = 0.0;
    if (!xy_speed_inited_) {
        xy_speed_inited_ = true;
        prev_pos_for_xy_vel_ = current_pos;
        prev_xy_vel_time_ = xy_now;
    } else {
        const double dt_xy = clampd((xy_now - prev_xy_vel_time_).toSec(), 0.005, 0.10);
        const Eigen::Vector3d dp = current_pos - prev_pos_for_xy_vel_;
        horiz_speed = std::hypot(dp.x(), dp.y()) / std::max(1e-3, dt_xy);
        prev_pos_for_xy_vel_ = current_pos;
        prev_xy_vel_time_ = xy_now;
    }

    const double lookahead_now = clampd(lookahead_dist_ +
                                        lookahead_time_gain_ * std::max(0.0, horiz_speed - lookahead_speed_ref_),
                                        lookahead_min_dist_,
                                        lookahead_max_dist_);
    const size_t i_tar = find_lookahead_index(i_near, lookahead_now);
    const Eigen::Vector3d p_tar_raw = path_pts_[i_tar];
    Eigen::Vector3d p_tar = p_tar_raw;
    const Eigen::Matrix3d R_wb = current_quat.normalized().toRotationMatrix();
    stage2_debug_candidate_targets_.clear();
    stage2_debug_candidate_scores_.clear();
    stage2_debug_candidate_clearances_.clear();
    stage2_debug_candidate_gap_widths_.clear();
    stage2_debug_selected_candidate_ = -1;
    stage2_debug_blockage_dist_ = std::numeric_limits<double>::infinity();
    stage2_debug_blockage_points_ = 0;
    stage2_debug_blockage_density_ = 0.0;
    stage2_debug_blockage_occupied_bins_ = 0;
    stage2_debug_best_candidate_clearance_ = std::numeric_limits<double>::infinity();
    stage2_debug_best_gap_width_ = 0.0;
    Stage2OccupancyGrid occ_grid;
    const bool have_occ_grid = stage2_avoid_enable_ && build_stage2_occupancy_grid(p_tar_raw, occ_grid);
    double stage2_avoid_offset = 0.0;
    if (have_occ_grid) {
        compute_stage2_blockage_metrics(occ_grid,
                                        stage2_debug_blockage_dist_,
                                        stage2_debug_blockage_points_,
                                        stage2_debug_blockage_density_,
                                        stage2_debug_blockage_occupied_bins_);
    }
    double stage2_obs_dist = stage2_debug_blockage_dist_;
    const bool stage2_blocked_raw =
        std::isfinite(stage2_obs_dist) &&
        (stage2_obs_dist < stage2_avoid_trigger_dist_);
    const ros::Time avoid_now = ros::Time::now();
    double closing_speed = 0.0;
    if (std::isfinite(stage2_obs_dist) && std::isfinite(stage2_prev_blockage_dist_) && !stage2_prev_blockage_time_.isZero()) {
        const double dt_obs = clampd((avoid_now - stage2_prev_blockage_time_).toSec(), 0.01, 0.20);
        closing_speed = std::max(0.0, (stage2_prev_blockage_dist_ - stage2_obs_dist) / dt_obs);
    }
    const double ttc = (closing_speed > 1e-3 && std::isfinite(stage2_obs_dist))
                           ? stage2_obs_dist / closing_speed
                           : std::numeric_limits<double>::infinity();
    const bool dynamic_threat =
        std::isfinite(stage2_obs_dist) &&
        (closing_speed >= stage2_dynamic_closing_speed_thresh_) &&
        (ttc <= stage2_dynamic_ttc_thresh_);
    stage2_prev_blockage_dist_ = stage2_obs_dist;
    stage2_prev_blockage_time_ = avoid_now;

    const bool stage2_blocked = stage2_blocked_raw || dynamic_threat;
    const bool stage2_triggered = stage2_blocked;
    stage2_triggered_streak_ = stage2_triggered ? (stage2_triggered_streak_ + 1) : 0;
    if (stage2_blocked) {
        stage2_avoid_latched_ = true;
        stage2_last_blocked_time_ = avoid_now;
    }
    const auto rejoin_segment_is_clear = [&](const Eigen::Vector3d& seg_start_w,
                                             const Eigen::Vector3d& seg_end_w) {
        if (!stage2_frame_cloud_body_ || stage2_frame_cloud_body_->points.empty()) return true;
        const Eigen::Vector3d seg_start_b = R_wb.transpose() * (seg_start_w - current_pos);
        const Eigen::Vector3d seg_end_b = R_wb.transpose() * (seg_end_w - current_pos);
        const double clearance_radius = std::max(0.25, 0.5 * stage2_vehicle_safe_width_);
        const double z_clearance = std::max(0.20, 0.5 * stage2_vehicle_safe_height_);
        const double x_min = std::min(seg_start_b.x(), seg_end_b.x()) - clearance_radius;
        const double x_max = std::max(seg_start_b.x(), seg_end_b.x()) + clearance_radius;
        const double y_min = std::min(seg_start_b.y(), seg_end_b.y()) - clearance_radius;
        const double y_max = std::max(seg_start_b.y(), seg_end_b.y()) + clearance_radius;
        const double z_min = std::min(seg_start_b.z(), seg_end_b.z()) - z_clearance;
        const double z_max = std::max(seg_start_b.z(), seg_end_b.z()) + z_clearance;
        for (const auto& p : stage2_frame_cloud_body_->points) {
            if (p.x < x_min || p.x > x_max || p.y < y_min || p.y > y_max || p.z < z_min || p.z > z_max) continue;
            const Eigen::Vector3d p_b(p.x, p.y, p.z);
            if (point_to_segment_distance(p_b, seg_start_b, seg_end_b) <= clearance_radius) {
                return false;
            }
        }
        return true;
    };
    const auto compute_vertical_and_body_clearance = [&]() {
        double overhead_min = std::numeric_limits<double>::infinity();
        double underside_min = std::numeric_limits<double>::infinity();
        double body_proximity_min = std::numeric_limits<double>::infinity();
        Eigen::Vector3d nearest_body_point_b = Eigen::Vector3d::Zero();
        bool nearest_body_point_valid = false;
        if (!stage2_frame_cloud_body_ || stage2_frame_cloud_body_->points.empty()) {
            return std::make_tuple(overhead_min, underside_min, body_proximity_min,
                                   nearest_body_point_b, nearest_body_point_valid);
        }
        const double lateral_band = std::max(0.35, 0.8 * stage2_vehicle_safe_width_);
        const double forward_band = std::max(0.8, 1.5 * stage2_vehicle_safe_width_);
        const double vertical_band = std::max(0.30, 0.8 * stage2_vehicle_safe_height_);
        const double body_eval_radius = std::max(0.8, 1.8 * stage2_vehicle_safe_width_);
        for (const auto& p : stage2_frame_cloud_body_->points) {
            const double abs_y = std::fabs(p.y);
            if (p.x >= -0.2 && p.x <= forward_band && abs_y <= lateral_band) {
                if (p.z > vertical_band) {
                    overhead_min = std::min(overhead_min, static_cast<double>(p.z));
                } else if (p.z < -vertical_band) {
                    underside_min = std::min(underside_min, static_cast<double>(-p.z));
                }
            }
            if (p.x >= -0.3 && p.x <= body_eval_radius && abs_y <= body_eval_radius) {
                const double body_dist = std::sqrt(static_cast<double>(p.x) * p.x +
                                                   static_cast<double>(p.y) * p.y +
                                                   static_cast<double>(p.z) * p.z);
                if (body_dist < body_proximity_min) {
                    body_proximity_min = body_dist;
                    nearest_body_point_b = Eigen::Vector3d(p.x, p.y, p.z);
                    nearest_body_point_valid = true;
                }
            }
        }
        return std::make_tuple(overhead_min, underside_min, body_proximity_min,
                               nearest_body_point_b, nearest_body_point_valid);
    };
    const auto [overhead_min, underside_min, body_proximity_min, nearest_body_point_b, nearest_body_point_valid] = compute_vertical_and_body_clearance();
    const bool overhead_clear = !std::isfinite(overhead_min) || overhead_min >= std::max(0.8, 1.5 * stage2_vehicle_safe_height_);
    const bool underside_clear = !std::isfinite(underside_min) || underside_min >= std::max(0.8, 1.5 * stage2_vehicle_safe_height_);
    const bool body_clearance_ok = !std::isfinite(body_proximity_min) || body_proximity_min >= std::max(1.0, 1.5 * stage2_vehicle_safe_width_);
    const bool rejoin_safe_now = rejoin_segment_is_clear(current_pos, p_tar_raw);
    const bool close_quarters_escape = !body_clearance_ok || !overhead_clear || !underside_clear;
    const bool obstacle_cleared =
        !stage2_blocked_raw &&
        !dynamic_threat &&
        (!std::isfinite(stage2_obs_dist) || stage2_obs_dist >= stage2_recover_clearance_) &&
        (stage2_debug_blockage_points_ < std::max(1, stage2_blockage_min_points_ / 2)) &&
        (stage2_debug_blockage_density_ < 0.5 * stage2_blockage_density_thresh_) &&
        overhead_clear &&
        underside_clear &&
        body_clearance_ok &&
        rejoin_safe_now;
    if (stage2_avoid_latched_) {
        const double clear_elapsed =
            stage2_last_blocked_time_.isZero() ? 0.0 : (avoid_now - stage2_last_blocked_time_).toSec();
        if (obstacle_cleared && clear_elapsed >= stage2_recover_hold_sec_) {
            stage2_avoid_latched_ = false;
            stage2_last_avoid_target_valid_ = false;
        }
    }
    const bool replan_active = stage2_blocked || stage2_avoid_latched_ || close_quarters_escape;
    // Only generate a fresh local target when the current frame says replanning is needed.
    const bool have_plan_target =
        replan_active && have_occ_grid && compute_stage2_avoid_target(p_tar_raw, occ_grid, p_tar, stage2_avoid_offset);
    bool have_avoid_target = have_plan_target;
    if (stage2_blocked || have_avoid_target || close_quarters_escape) {
        stage2_mode_ = Stage2FollowMode::AVOID;
    }

    if (have_avoid_target) {
        stage2_last_avoid_target_ = p_tar;
        stage2_last_avoid_target_valid_ = true;
    }
    if (dynamic_threat && have_avoid_target) {
        const Eigen::Vector3d raw_target_b_now = R_wb.transpose() * (p_tar_raw - current_pos);
        Eigen::Vector3d avoid_target_b_now = R_wb.transpose() * (p_tar - current_pos);
        const double max_dynamic_offset = std::max(0.1, stage2_dynamic_avoid_max_lateral_offset_);
        const double limited_y = clampd(avoid_target_b_now.y(),
                                        raw_target_b_now.y() - max_dynamic_offset,
                                        raw_target_b_now.y() + max_dynamic_offset);
        avoid_target_b_now.y() = limited_y;
        const double max_dynamic_dz = std::max(0.3, 0.5 * stage2_corridor_half_height_safe_);
        avoid_target_b_now.z() = clampd(avoid_target_b_now.z(),
                                        raw_target_b_now.z() - max_dynamic_dz,
                                        raw_target_b_now.z() + max_dynamic_dz);
        p_tar = current_pos + R_wb * avoid_target_b_now;
        stage2_avoid_offset = limited_y - raw_target_b_now.y();
    }

    if (have_avoid_target) {
        // Slew-limit the local target so the controller follows a continuous path.
        const double target_dt = 0.03;
        if (!stage2_smoothed_avoid_target_valid_) {
            stage2_smoothed_avoid_target_ = p_tar;
            stage2_smoothed_avoid_target_valid_ = true;
        } else {
            Eigen::Vector3d delta_w = p_tar - stage2_smoothed_avoid_target_;
            Eigen::Vector2d delta_xy = delta_w.head<2>();
            const double max_xy_step = std::max(0.1, stage2_avoid_target_xy_slew_rate_) * target_dt;
            const double delta_xy_norm = delta_xy.norm();
            if (delta_xy_norm > max_xy_step && delta_xy_norm > 1e-6) {
                delta_xy *= (max_xy_step / delta_xy_norm);
            }
            const double max_z_step = std::max(0.05, stage2_avoid_target_z_slew_rate_) * target_dt;
            delta_w.z() = clampd(delta_w.z(), -max_z_step, max_z_step);
            stage2_smoothed_avoid_target_.head<2>() += delta_xy;
            stage2_smoothed_avoid_target_.z() += delta_w.z();
        }
        p_tar = stage2_smoothed_avoid_target_;
    } else {
        stage2_smoothed_avoid_target_valid_ = false;
        stage2_last_candidate_yaw_idx_ = -1;
        stage2_last_candidate_pitch_idx_ = -1;
    }

    if (have_avoid_target) {
        stage2_local_traj_.clear();
        stage2_local_traj_.push_back(current_pos);
        const double spacing = std::max(0.2, stage2_local_traj_spacing_);
        Eigen::Vector3d avoid_dir = p_tar - current_pos;
        const double dir_norm = avoid_dir.norm();
        if (dir_norm > 1e-3) {
            avoid_dir /= dir_norm;
        } else {
            avoid_dir = Eigen::Vector3d::UnitX();
        }
        const double forward_horizon =
            std::max(stage2_local_traj_lookahead_dist_, std::min(lookahead_now, stage2_avoid_forward_range_));
        const int forward_steps = std::max(1, static_cast<int>(std::ceil(forward_horizon / spacing)));
        for (int s = 1; s <= forward_steps; ++s) {
            const double travel = std::min(forward_horizon, static_cast<double>(s) * spacing);
            const Eigen::Vector3d traj_pt = current_pos + travel * avoid_dir;
            stage2_local_traj_.push_back(traj_pt);
        }
        const auto segment_is_clear = [&](const Eigen::Vector3d& seg_start_w,
                                          const Eigen::Vector3d& seg_end_w) {
            if (!stage2_frame_cloud_body_ || stage2_frame_cloud_body_->points.empty()) return true;
            const Eigen::Vector3d seg_start_b = R_wb.transpose() * (seg_start_w - current_pos);
            const Eigen::Vector3d seg_end_b = R_wb.transpose() * (seg_end_w - current_pos);
            const double clearance_radius =
                std::max(0.25, 0.5 * stage2_vehicle_safe_width_);
            const double z_clearance =
                std::max(0.20, 0.5 * stage2_vehicle_safe_height_);
            const double x_min = std::min(seg_start_b.x(), seg_end_b.x()) - clearance_radius;
            const double x_max = std::max(seg_start_b.x(), seg_end_b.x()) + clearance_radius;
            const double y_min = std::min(seg_start_b.y(), seg_end_b.y()) - clearance_radius;
            const double y_max = std::max(seg_start_b.y(), seg_end_b.y()) + clearance_radius;
            const double z_min = std::min(seg_start_b.z(), seg_end_b.z()) - z_clearance;
            const double z_max = std::max(seg_start_b.z(), seg_end_b.z()) + z_clearance;
            for (const auto& p : stage2_frame_cloud_body_->points) {
                if (p.x < x_min || p.x > x_max || p.y < y_min || p.y > y_max || p.z < z_min || p.z > z_max) continue;
                const Eigen::Vector3d p_b(p.x, p.y, p.z);
                if (point_to_segment_distance(p_b, seg_start_b, seg_end_b) <= clearance_radius) {
                    return false;
                }
            }
            return true;
        };

        const Eigen::Vector3d rejoin = p_tar_raw;
        if ((rejoin - stage2_local_traj_.back()).norm() > 0.3 &&
            segment_is_clear(stage2_local_traj_.back(), rejoin)) {
            // Only reconnect to the global centerline if the reconnect segment is clear.
            const Eigen::Vector3d seg2 = rejoin - stage2_local_traj_.back();
            const double seg2_len = seg2.norm();
            const int steps2 = std::max(1, static_cast<int>(std::ceil(seg2_len / spacing)));
            const Eigen::Vector3d start2 = stage2_local_traj_.back();
            for (int s = 1; s <= steps2; ++s) {
                const double alpha = static_cast<double>(s) / static_cast<double>(steps2);
                stage2_local_traj_.push_back(start2 + alpha * seg2);
            }
        }
    }

    bool have_local_tangent = false;
    Eigen::Vector3d local_tangent_w = R_wb.col(0);
    if (stage2_mode_ == Stage2FollowMode::AVOID && stage2_local_traj_.size() >= 2) {
        const size_t local_near = find_nearest_index_in_points(stage2_local_traj_, current_pos);
        const size_t local_tar =
            find_lookahead_index_in_points(stage2_local_traj_, local_near, std::max(0.5, stage2_local_traj_lookahead_dist_));
        p_tar = stage2_local_traj_[local_tar];
        // Drive the vehicle along the local path tangent instead of repeatedly pointing at a single point.
        const size_t local_next = std::min(stage2_local_traj_.size() - 1, local_tar + 1);
        if (local_next > local_tar) {
            local_tangent_w = stage2_local_traj_[local_next] - stage2_local_traj_[local_tar];
            local_tangent_w.z() = 0.0;
            if (local_tangent_w.head<2>().norm() > 1e-3) {
                local_tangent_w.normalize();
                have_local_tangent = true;
            }
        }
    } else {
        stage2_local_traj_.clear();
    }
    publish_target_marker(p_tar);

    const Eigen::Vector3d e_w = p_tar - current_pos;
    const Eigen::Vector3d e_b = R_wb.transpose() * e_w;

    Eigen::Vector3d yaw_ref_b = e_b;
    if (have_local_tangent) {
        yaw_ref_b = R_wb.transpose() * local_tangent_w;
    }
    const double yaw_err = std::atan2(yaw_ref_b.y(), std::max(1e-3, yaw_ref_b.x()));
    const double yaw_err_deg = yaw_err * 180.0 / M_PI;
    const ros::Time yaw_now = ros::Time::now();
    double yaw_rate_meas_ccw = 0.0;
    const double yaw_cur_rad = quat_to_yaw_rad(current_quat);
    if (!yaw_rate_inited_) {
        yaw_rate_inited_ = true;
        yaw_prev_rad_ = yaw_cur_rad;
        yaw_prev_time_ = yaw_now;
        yaw_rate_meas_filt_ = 0.0;
    } else {
        const double dt_yaw = clampd((yaw_now - yaw_prev_time_).toSec(), 0.005, 0.10);
        const double yaw_delta = wrap_pi_rad(yaw_cur_rad - yaw_prev_rad_);
        const double yaw_rate_raw = yaw_delta * 180.0 / M_PI / std::max(1e-3, dt_yaw);
        yaw_rate_meas_filt_ =
            (1.0 - yaw_rate_lpf_alpha_) * yaw_rate_meas_filt_ + yaw_rate_lpf_alpha_ * yaw_rate_raw;
        yaw_rate_meas_ccw = yaw_rate_meas_filt_;
        yaw_prev_rad_ = yaw_cur_rad;
        yaw_prev_time_ = yaw_now;
    }

    if (std::fabs(yaw_err_deg) <= yaw_stop_tolerance_deg_) {
        yaw_deg = 0.0;
    } else {
        yaw_deg = fp_kp_yaw_ * yaw_err_deg + fp_kd_yaw_rate_ * yaw_rate_meas_ccw;
        yaw_deg = clampd(yaw_deg, -yaw_soft_limit_deg_, yaw_soft_limit_deg_);
        if (std::fabs(yaw_deg) < yaw_min_cmd_deg_) {
            yaw_deg = (yaw_err_deg >= 0.0) ? yaw_min_cmd_deg_ : -yaw_min_cmd_deg_;
        }
    }

    const bool dynamic_fast_avoid = dynamic_threat;
    const double stage2_kp_y_scale_now =
        (stage2_mode_ == Stage2FollowMode::AVOID)
            ? (dynamic_fast_avoid ? stage2_dynamic_avoid_vy_gain_scale_ : stage2_avoid_vy_gain_scale_)
            : 1.0;
    const double stage2_kp_y_now = fp_kp_y_ * stage2_kp_y_scale_now;
    const double stage2_vy_limit_now =
        (stage2_mode_ == Stage2FollowMode::AVOID)
            ? (dynamic_fast_avoid ? stage2_dynamic_avoid_max_vy_ : stage2_avoid_max_vy_)
            : stage2_max_vy_;
    const double stage2_vz_limit_now =
        (stage2_mode_ == Stage2FollowMode::AVOID)
            ? (dynamic_fast_avoid ? stage2_dynamic_avoid_max_vz_ : stage2_avoid_max_vz_)
            : stage2_max_vz_;
    if (stage2_mode_ == Stage2FollowMode::AVOID && have_local_tangent) {
        Eigen::Vector3d tangent_b = R_wb.transpose() * local_tangent_w;
        tangent_b.z() = 0.0;
        const double tangent_norm = tangent_b.head<2>().norm();
        if (tangent_norm > 1e-3) {
            tangent_b.head<2>() /= tangent_norm;
        } else {
            tangent_b = Eigen::Vector3d::UnitX();
        }
        const double track_speed =
            clampd(std::max(2.0, e_w.head<2>().norm()), 2.0, stage2_max_vx_);
        const double vx_ff = std::max(0.0, track_speed * tangent_b.x());
        const double vy_ff = track_speed * tangent_b.y();
        vx = clampd(vx_ff + 0.6 * fp_kp_x_ * e_b.x(), -stage2_max_vx_, stage2_max_vx_);
        vy = clampd(vy_ff + stage2_kp_y_now * e_b.y(), -stage2_vy_limit_now, stage2_vy_limit_now);
    } else {
        vx = clampd(fp_kp_x_ * e_b.x(), -stage2_max_vx_, stage2_max_vx_);
        vy = clampd(stage2_kp_y_now * e_b.y(), -stage2_vy_limit_now, stage2_vy_limit_now);
    }

    const double ey_abs = std::fabs(e_b.y());
    double vx_lateral_scale = 1.0;
    double lateral_thresh_now = lateral_slow_err_thresh_;
    const ros::Time t_now_for_gate = ros::Time::now();
    if (!takeoff_success_time.isZero()) {
        const double flight_time = (t_now_for_gate - takeoff_success_time).toSec();
        if (flight_time >= gate_relax_time_sec_) {
            lateral_thresh_now = late_lateral_err_thresh_;
        }
    }
    if (ey_abs > lateral_thresh_now) {
        const double ey_excess = ey_abs - lateral_thresh_now;
        vx_lateral_scale = clampd(1.0 / (1.0 + lateral_slow_gain_ * ey_excess), lateral_slow_min_scale_, 1.0);
    }
    vx *= vx_lateral_scale;

    const double vy_alpha = clampd(fp_vy_lpf_alpha_, 0.0, 1.0);
    if (!xy_vel_inited_) {
        vy_cmd_filt_ = vy;
        xy_vel_inited_ = true;
    } else if (ey_abs < 4.0) {
        vy_cmd_filt_ = (1.0 - vy_alpha) * vy_cmd_filt_ + vy_alpha * vy;
        vy = vy_cmd_filt_;
    } else {
        vy_cmd_filt_ = vy;
    }
    if (stage2_mode_ == Stage2FollowMode::AVOID && have_avoid_target) {
        const double lateral_intent = std::fabs(e_b.y());
        if (lateral_intent > 0.30) {
            const double min_avoid_vy =
                std::min(stage2_vy_limit_now, stage2_avoid_min_vy_cmd_ + 0.8 * lateral_intent);
            if (std::fabs(vy) < min_avoid_vy) {
                vy = std::copysign(min_avoid_vy, e_b.y());
                vy_cmd_filt_ = vy;
            }
        }
    }

    // [P0-边界约束] 横向回拉：基于"当前位置偏离中心线"的硬约束，防止避障时冲出赛道。
    // 关键：之前的 avoid_offset 只约束目标点偏移，不约束实际位置；这里补上基于 current_pos 的反馈。
    {
        const size_t seg_a = i_near;
        const size_t seg_b = std::min(path_pts_.size() - 1, i_near + 1);
        Eigen::Vector3d tan_w = (seg_b > seg_a) ? Eigen::Vector3d(path_pts_[seg_b] - path_pts_[seg_a])
                                                 : Eigen::Vector3d(Eigen::Vector3d::UnitX());
        tan_w.z() = 0.0;
        if (tan_w.head<2>().norm() < 1e-3) tan_w = Eigen::Vector3d::UnitX(); else tan_w.normalize();
        // 中心线左法向(世界系)
        Eigen::Vector3d nrm_w(-tan_w.y(), tan_w.x(), 0.0);
        Eigen::Vector3d err_w = current_pos - path_pts_[seg_a];
        err_w.z() = 0.0;
        const double signed_dev = err_w.dot(nrm_w);          // 带符号横向偏离(+左 -右)
        const double abs_dev = std::fabs(signed_dev);
        const double hard_lim = std::max(0.5, stage2_corridor_half_width_safe_);
        const double soft_lim = clampd(stage2_corridor_pull_soft_ratio_, 0.1, 0.95) * hard_lim;
        if (abs_dev > soft_lim) {
            // 回拉强度随偏离二次增长，硬边界处饱和
            const double over = clampd((abs_dev - soft_lim) / std::max(1e-3, hard_lim - soft_lim), 0.0, 1.0);
            const double pull_mag = clampd(stage2_corridor_pull_gain_ * over * over * stage2_corridor_pull_max_vy_,
                                           0.0, stage2_corridor_pull_max_vy_);
            // 回拉方向：朝中心线(signed_dev>0在左侧→需朝右拉=-nrm_w)，转到机体系取y分量
            const Eigen::Vector3d pull_dir_w = (signed_dev > 0.0 ? -nrm_w : nrm_w);
            const double pull_vy_b = (R_wb.transpose() * pull_dir_w).y() * pull_mag;
            vy = clampd(vy + pull_vy_b, -stage2_vy_limit_now, stage2_vy_limit_now);
            vy_cmd_filt_ = vy;
            // 趋硬边界时压低前向：宁可慢，不出界，留时间拉回
            const double vx_scale = clampd(1.0 - (1.0 - stage2_corridor_hard_vx_scale_) * over,
                                           stage2_corridor_hard_vx_scale_, 1.0);
            vx *= vx_scale;
        }
    }

    const size_t i_tar_z = find_lookahead_index(i_near, std::max(0.0, z_lookahead_dist_));
    double z_target = path_pts_[i_tar_z].z();
    const double stage2_travel_xy =
        stage2_start_pos_valid_ ? (current_pos.head<2>() - stage2_start_pos_.head<2>()).norm() : 0.0;
    if (stage2_avoid_latched_) {
        stage2_z_blend_pause_xy_ += std::max(0.0, stage2_travel_xy - stage2_prev_travel_xy_);
    }
    stage2_prev_travel_xy_ = stage2_travel_xy;
    const double stage2_effective_travel_xy = std::max(0.0, stage2_travel_xy - stage2_z_blend_pause_xy_);
    const double blend_dist = std::max(0.1, stage2_z_blend_dist_);
    const double stage2_z_blend_alpha =
        clampd(stage2_effective_travel_xy / blend_dist,
               0.0, 1.0);
    z_target = (1.0 - stage2_z_blend_alpha) * stage2_hold_z_ + stage2_z_blend_alpha * z_target;
    const ros::Time z_now = ros::Time::now();
    double dt = 0.03;
    if (z_pid_inited_) {
        dt = clampd((z_now - z_pid_prev_time_).toSec(), 0.005, 0.10);
    } else {
        z_pid_int_ = 0.0;
        z_pid_prev_err_ = 0.0;
        z_pid_d_filt_ = 0.0;
        z_pid_inited_ = true;
    }
    const bool stage2_z_avoid_active =
        (stage2_mode_ == Stage2FollowMode::AVOID) &&
        have_avoid_target &&
        (stage2_triggered_streak_ >= std::max(1, stage2_z_avoid_trigger_frames_));
    // Keep z on a single logic while stage2 is still resolving an obstacle:
    // either follow an explicit vertical avoid target, or hold the current altitude.
    const bool stage2_z_hold_active =
        (((stage2_mode_ == Stage2FollowMode::AVOID) || stage2_avoid_latched_) &&
         !stage2_z_avoid_active) ||
        (!rejoin_safe_now && !stage2_z_avoid_active) ||
        close_quarters_escape;
    if (stage2_z_avoid_active) {
        z_target = p_tar.z();
    } else if (stage2_z_hold_active) {
        // Default to holding the current altitude until the surrounding clearance stays safe.
        z_target = current_pos.z();
    }
    const double dz_w = z_target - current_pos.z();

    if (stage2_z_hold_active && !stage2_z_avoid_active) {
        z_pid_int_ = 0.0;
        z_pid_prev_err_ = 0.0;
        z_pid_d_filt_ = 0.0;
        z_prev_meas_ = current_pos.z();
        z_prev_ref_ = current_pos.z();
        z_vel_meas_filt_ = 0.0;
        z_ref_vel_filt_ = 0.0;
        z_vel_inited_ = true;
        vz = 0.0;
    } else {
        if (!z_vel_inited_) {
            z_prev_meas_ = current_pos.z();
            z_prev_ref_ = z_target;
            z_vel_meas_filt_ = 0.0;
            z_ref_vel_filt_ = 0.0;
            z_vel_inited_ = true;
        }
        const double v_meas_raw = (current_pos.z() - z_prev_meas_) / std::max(1e-3, dt);
        z_vel_meas_filt_ =
            (1.0 - z_vel_lpf_alpha_) * z_vel_meas_filt_ + z_vel_lpf_alpha_ * v_meas_raw;
        z_prev_meas_ = current_pos.z();
        const double v_ref_raw = (z_target - z_prev_ref_) / std::max(1e-3, dt);
        z_ref_vel_filt_ =
            (1.0 - z_ref_vel_lpf_alpha_) * z_ref_vel_filt_ + z_ref_vel_lpf_alpha_ * v_ref_raw;
        z_prev_ref_ = z_target;
        const double v_ref = clampd(z_ref_vel_ff_gain_ * z_ref_vel_filt_, -max_vz_down_, max_vz_);
        const double v_err = v_ref - z_vel_meas_filt_;
        if ((dz_w > 0.0 && z_pid_int_ < 0.0) || (dz_w < 0.0 && z_pid_int_ > 0.0)) {
            z_pid_int_ *= 0.5;
        }
        z_pid_d_filt_ = (1.0 - z_pid_d_lpf_alpha_) * z_pid_d_filt_ + z_pid_d_lpf_alpha_ * v_err;
        const double int_candidate = clampd(z_pid_int_ + dz_w * dt, -z_pid_i_limit_, z_pid_i_limit_);
        const double u_unsat_candidate = z_pid_kp_ * dz_w + z_pid_ki_ * int_candidate + z_pid_kd_ * z_pid_d_filt_;
        const double u_sat_candidate = clampd(u_unsat_candidate, -max_vz_down_, max_vz_);
        const bool allow_integrate =
            (std::fabs(u_unsat_candidate - u_sat_candidate) < 1e-6) ||
            ((u_unsat_candidate > u_sat_candidate) && (dz_w < 0.0)) ||
            ((u_unsat_candidate < u_sat_candidate) && (dz_w > 0.0));
        if (allow_integrate) {
            z_pid_int_ = int_candidate;
        }
        const double z_gain_scale =
            (stage2_mode_ == Stage2FollowMode::AVOID)
                ? (dynamic_fast_avoid ? stage2_dynamic_avoid_vz_gain_scale_ : stage2_avoid_vz_gain_scale_)
                : 1.0;
        const double u_unsat = z_gain_scale * (z_pid_kp_ * dz_w + z_pid_ki_ * z_pid_int_ + z_pid_kd_ * z_pid_d_filt_);
        vz = clampd(u_unsat, -stage2_vz_limit_now, stage2_vz_limit_now);
        if (!stage2_z_avoid_active && !stage2_z_hold_active) {
            const double recover_vz_cap = std::max(0.2, stage2_recover_max_vz_);
            vz = clampd(vz, -recover_vz_cap, recover_vz_cap);
        }
    }
    z_pid_prev_err_ = dz_w;
    z_pid_prev_time_ = z_now;

    if (enable_yaw_only_gate_ &&
        stage2_mode_ != Stage2FollowMode::AVOID &&
        std::fabs(yaw_err_deg) > yaw_only_deg_) {
        const double scale_vx = clampd(yaw_only_deg_ / std::fabs(yaw_err_deg), 0.20, 1.0);
        const double scale_vy = clampd(std::sqrt(scale_vx), 0.55, 1.0);
        vx *= scale_vx;
        vy *= scale_vy;
    }

    const double dist_end = (path_pts_.back() - current_pos).norm();
    const double stage2_avoid_scale = 1.0;

    // [METRIC] 横向偏离中心线：current_pos 到最近路径段的水平垂距（对应"被吸下去"前兆）
    double centerline_dev = 0.0;
    {
        const size_t seg_a = i_near;
        const size_t seg_b = std::min(path_pts_.size() - 1, i_near + 1);
        Eigen::Vector3d p_flat = current_pos;        p_flat.z() = 0.0;
        Eigen::Vector3d a_flat = path_pts_[seg_a];   a_flat.z() = 0.0;
        Eigen::Vector3d b_flat = path_pts_[seg_b];   b_flat.z() = 0.0;
        centerline_dev = (seg_b > seg_a)
            ? point_to_segment_distance(p_flat, a_flat, b_flat)
            : (p_flat - a_flat).norm();
    }

    const double dt_cmd = clampd(dt, 0.01, 0.10);
    const double vy_slew_rate =
        dynamic_fast_avoid ? stage2_dynamic_avoid_vy_slew_rate_ : stage2_avoid_vy_slew_rate_;
    const double vz_slew_rate =
        dynamic_fast_avoid ? stage2_dynamic_avoid_vz_slew_rate_ : stage2_avoid_vz_slew_rate_;
    const double max_dvy = std::max(0.0, vy_slew_rate) * dt_cmd;
    const double max_dvz = std::max(0.0, vz_slew_rate) * dt_cmd;
    if (!stage2_cmd_inited_) {
        stage2_prev_vy_cmd_ = vy;
        stage2_prev_vz_cmd_ = vz;
        stage2_cmd_inited_ = true;
    } else {
        vy = clampd(vy,
                    stage2_prev_vy_cmd_ - max_dvy,
                    stage2_prev_vy_cmd_ + max_dvy);
        vz = clampd(vz,
                    stage2_prev_vz_cmd_ - max_dvz,
                    stage2_prev_vz_cmd_ + max_dvz);
        stage2_prev_vy_cmd_ = vy;
        stage2_prev_vz_cmd_ = vz;
    }

    publish_stage2_obstacle_points();
    publish_stage2_debug_markers(p_tar_raw, p_tar, stage2_obs_dist, stage2_avoid_offset, stage2_avoid_scale);

    if (stage2_mode_ == Stage2FollowMode::CRUISE && !stage2_avoid_latched_ &&
        dist_end < final_stop_before_end_dist_) finished = true;

    ROS_INFO_THROTTLE(0.5, "[STAGE2_PATH] mode=%s near=%zu tar=%zu tar_z=%zu L=%.2f vh=%.2f yaw_err=%.1f dz=%.2f zblend=%.2f obs=%.2f trigger=%.2f trig=%d trig_n=%d ztrig=%d head=%.2f foot=%.2f body=%.2f close=%d pts=%d dens=%.2f bins=%d gap=%d best=%.2f gapw=%.2f closing=%.2f ttc=%.2f latched=%d dyn=%d cdev=%.2f feas=%d tot=%d fb=%d cmd=(%.2f %.2f %.2f %.1f) lim(vy,vz)=(%.2f,%.2f)",
                      stage2_mode_name(stage2_mode_), i_near, i_tar, i_tar_z, lookahead_now, horiz_speed, yaw_err_deg, dz_w,
                      stage2_z_blend_alpha, stage2_obs_dist, stage2_avoid_trigger_dist_, static_cast<int>(stage2_triggered), stage2_triggered_streak_, static_cast<int>(stage2_z_avoid_active), overhead_min, underside_min, body_proximity_min, static_cast<int>(close_quarters_escape), stage2_debug_blockage_points_, stage2_debug_blockage_density_, stage2_debug_blockage_occupied_bins_,
                      static_cast<int>(have_avoid_target), stage2_debug_best_candidate_clearance_, stage2_debug_best_gap_width_,
                      closing_speed, ttc, static_cast<int>(stage2_avoid_latched_), static_cast<int>(dynamic_fast_avoid),
                      centerline_dev,
                      stage2_debug_feasible_candidates_, stage2_debug_total_candidates_, static_cast<int>(stage2_debug_fallback_used_),
                      vx, vy, vz, yaw_deg, stage2_vy_limit_now, stage2_vz_limit_now);
}

void BasicDev::publish_desired_path_once()
{
    if (!path_loaded_ || path_pts_.size() < 2) return;
    if (!desired_path_msg_.poses.empty()) {
        desired_path_msg_.header.stamp = ros::Time::now();
        desired_path_msg_.header.frame_id = viz_frame_id_;
        desired_path_pub_.publish(desired_path_msg_);
        return;
    }

    desired_path_msg_.poses.clear();
    desired_path_msg_.header.frame_id = viz_frame_id_;
    desired_path_msg_.header.stamp = ros::Time::now();
    desired_path_msg_.poses.reserve(path_pts_.size());

    for (const auto& p : path_pts_) {
        geometry_msgs::PoseStamped ps;
        ps.header.frame_id = viz_frame_id_;
        ps.header.stamp = ros::Time::now();
        ps.pose.position.x = p.x();
        ps.pose.position.y = p.y();
        ps.pose.position.z = p.z();
        ps.pose.orientation.x = 0.0;
        ps.pose.orientation.y = 0.0;
        ps.pose.orientation.z = 0.0;
        ps.pose.orientation.w = 1.0;
        desired_path_msg_.poses.push_back(ps);
    }
    desired_path_pub_.publish(desired_path_msg_);
}

// -----------------------------------------------------------------------------
// RViz / command publishing helpers
// -----------------------------------------------------------------------------
void BasicDev::update_and_publish_actual_path()
{
    if (!has_gps) return;

    if (!rviz_inited_) {
        actual_path_msg_.poses.clear();
        actual_path_msg_.header.frame_id = viz_frame_id_;
        actual_path_msg_.header.stamp = ros::Time::now();
        rviz_inited_ = true;
        last_actual_pos_valid_ = false;
    }

    const Eigen::Vector3d p = current_pos;
    bool append = false;
    if (!last_actual_pos_valid_) {
        append = true;
    } else if ((p - last_actual_pos_).norm() >= actual_append_dist_) {
        append = true;
    }

    if (append) {
        geometry_msgs::PoseStamped ps;
        ps.header.stamp = ros::Time::now();
        ps.header.frame_id = viz_frame_id_;
        ps.pose.position.x = p.x();
        ps.pose.position.y = p.y();
        ps.pose.position.z = p.z();
        ps.pose.orientation.x = current_quat.x();
        ps.pose.orientation.y = current_quat.y();
        ps.pose.orientation.z = current_quat.z();
        ps.pose.orientation.w = current_quat.w();
        actual_path_msg_.poses.push_back(ps);
        last_actual_pos_ = p;
        last_actual_pos_valid_ = true;
    }

    if (actual_path_msg_.poses.size() > actual_max_points_) {
        const size_t erase_n = std::min<size_t>(500, actual_path_msg_.poses.size() / 2);
        actual_path_msg_.poses.erase(actual_path_msg_.poses.begin(),
                                     actual_path_msg_.poses.begin() + static_cast<long>(erase_n));
    }

    actual_path_msg_.header.stamp = ros::Time::now();
    actual_path_msg_.header.frame_id = viz_frame_id_;
    actual_path_pub_.publish(actual_path_msg_);
}

void BasicDev::publish_target_marker(const Eigen::Vector3d& p_tar)
{
    visualization_msgs::Marker m;
    m.header.stamp = ros::Time::now();
    m.header.frame_id = viz_frame_id_;
    m.ns = "target_point";
    m.id = 0;
    m.type = visualization_msgs::Marker::SPHERE;
    m.action = visualization_msgs::Marker::ADD;
    m.pose.position.x = p_tar.x();
    m.pose.position.y = p_tar.y();
    m.pose.position.z = p_tar.z();
    m.pose.orientation.x = 0.0;
    m.pose.orientation.y = 0.0;
    m.pose.orientation.z = 0.0;
    m.pose.orientation.w = 1.0;
    m.scale.x = 0.25;
    m.scale.y = 0.25;
    m.scale.z = 0.25;
    m.color.a = 1.0;
    m.color.r = 1.0;
    m.color.g = 0.0;
    m.color.b = 0.0;
    target_marker_pub_.publish(m);
}

void BasicDev::publish_follow_tf()
{
    if (!publish_follow_tf_ || !has_gps) return;

    geometry_msgs::TransformStamped tf_msg;
    tf_msg.header.stamp = ros::Time::now();
    tf_msg.header.frame_id = viz_frame_id_;
    tf_msg.child_frame_id = follow_frame_id_;
    tf_msg.transform.translation.x = current_pos.x();
    tf_msg.transform.translation.y = current_pos.y();
    tf_msg.transform.translation.z = current_pos.z();
    tf_msg.transform.rotation.x = current_quat.x();
    tf_msg.transform.rotation.y = current_quat.y();
    tf_msg.transform.rotation.z = current_quat.z();
    tf_msg.transform.rotation.w = current_quat.w();
    tf_broadcaster_.sendTransform(tf_msg);
}

void BasicDev::publish_velocity_cmd(double vx, double vy, double vz, double yaw_rate, uint8_t stop)
{
    velcmd.header.stamp = ros::Time::now();
    velcmd.vx = vx;
    velcmd.vy = vy;
    velcmd.vz = vz;
    velcmd.yawRate = yaw_rate;
    velcmd.va =10; // 固定值，暂不使用
    velcmd.stop = stop;
    vel_publisher.publish(velcmd);
}

// -----------------------------------------------------------------------------
// Main state machine and sensor callbacks
// -----------------------------------------------------------------------------
void BasicDev::control_timer_cb(const ros::TimerEvent& event)
{
    (void)event;
    // [PERF] 控制环耗时与帧率埋点（纯观测，不改控制逻辑）
    const ros::WallTime perf_loop_start = ros::WallTime::now();
    static ros::WallTime perf_prev_start;
    static bool perf_inited = false;
    static double perf_fps_filt = 0.0;       // 滑动平均帧率(Hz)
    static double perf_cb_ms_filt = 0.0;     // 滑动平均单帧耗时(ms)
    double perf_interval_ms = 0.0;
    if (perf_inited) {
        perf_interval_ms = (perf_loop_start - perf_prev_start).toSec() * 1000.0;
        const double inst_fps = (perf_interval_ms > 1e-3) ? (1000.0 / perf_interval_ms) : 0.0;
        perf_fps_filt = (perf_fps_filt <= 0.0) ? inst_fps : (0.9 * perf_fps_filt + 0.1 * inst_fps);
    }
    perf_prev_start = perf_loop_start;
    perf_inited = true;

    const ros::Time now = ros::Time::now();
    update_and_publish_actual_path();
    publish_follow_tf();

    if (!has_gps || !has_init_pose) {
        publish_velocity_cmd(0.0, 0.0, 0.0, 0.0, 0);
        ROS_INFO_THROTTLE(1.0, "等待数据: gps=%d init=%d",
            (int)has_gps, (int)has_init_pose);
        return;
    }

    // 阶段1：起飞处理（可配置是否调用服务）
    if (!has_takeoff) {
        if (!use_takeoff_service_) {
            has_takeoff = true;
            takeoff_success_time = now;
            enter_phase(TestPhase::HOVER_AFTER_TAKEOFF, now);
            ROS_INFO("已跳过起飞服务，直接进入控制流程。");
            return;
        }

        publish_velocity_cmd(0.0, 0.0, 0.0, 0.0, 0);
        if ((now - last_takeoff_try).toSec() > 1.0) {
            last_takeoff_try = now;
            if (takeoff_client.exists() || takeoff_client.waitForExistence(ros::Duration(0.2))) {
                const bool called = takeoff_client.call(takeoff);
                if (called && takeoff.response.success) {
                    has_takeoff = true;
                    takeoff_success_time = now;
                    enter_phase(TestPhase::HOVER_AFTER_TAKEOFF, now);
                    ROS_INFO("起飞成功，开始顺序动作测试。");
                } else {
                    ROS_WARN("起飞调用失败或success=false，继续重试。");
                }
            } else {
                ROS_WARN("起飞服务未就绪，继续重试。");
            }
        }
        return;
    }

    const double elapsed = phase_elapsed(now);

    double vx_cmd = 0.0;
    double vy_cmd = 0.0;
    double vz_cmd = 0.0;
    double yaw_rate_cmd = 0.0;
    bool publish_cmd = true;

    switch (phase) {
    case TestPhase::WAITING_DATA:
        enter_phase(TestPhase::HOVER_AFTER_TAKEOFF, now);
        break;
    case TestPhase::HOVER_AFTER_TAKEOFF:
        if (elapsed >= test_hover_sec) {
            enter_phase(TestPhase::MOVE_FORWARD_BUFFER, now);
        }
        break;
    case TestPhase::MOVE_FORWARD_BUFFER: {
        const double vx_forward = std::fabs(buffer_vx_);
        const double buffer_time = std::fabs(buffer_dist_) / std::max(0.1, vx_forward);
        const double rel_climb = current_pos.z() - init_pos.z();
        vx_cmd = vx_forward;
        vy_cmd = 0.0;
        vz_cmd = (rel_climb >= buffer_max_climb_ && buffer_vz_ > 0.0) ? 0.0 : buffer_vz_;
        yaw_rate_cmd = 0.0;
        if (elapsed >= buffer_time) {
            ROS_INFO("斜向缓冲完成: elapsed=%.2f s, 约前飞=%.2f m, 设定vz=%.2f m/s, rel_climb=%.2f m", elapsed, vx_forward * elapsed, buffer_vz_, rel_climb);
            if (buffer_hover_sec_ > 1e-3) {
                enter_phase(TestPhase::HOVER_AFTER_BUFFER, now);
            } else {
                if (!path_loaded_) {
                    if (!load_path_from_csv(current_pos)) {
                        ROS_ERROR("路径加载失败，进入FINISH悬停。");
                        enter_phase(TestPhase::FINISH, now);
                        break;
                    }
                }
                publish_desired_path_once();
                enter_phase(TestPhase::FOLLOW_PATH, now);
            }
        }
        break;
    }
    case TestPhase::HOVER_AFTER_BUFFER:
        if (elapsed >= buffer_hover_sec_) {
            if (!path_loaded_) {
                if (!load_path_from_csv(current_pos)) {
                    ROS_ERROR("路径加载失败，进入FINISH悬停。");
                    enter_phase(TestPhase::FINISH, now);
                    break;
                }
            }
            publish_desired_path_once();
            enter_phase(TestPhase::FOLLOW_PATH, now);
        }
        break;
    case TestPhase::FOLLOW_PATH: {
        if (!has_gps) {
            ROS_WARN_THROTTLE(1.0, "FOLLOW_PATH等待GPS初始化。");
            break;
        }
        bool finished = false;
        path_follow_step(vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd, finished);
        if (finished) {
            enter_phase(TestPhase::FINAL_HOVER_TURN, now);
        }
        break;
    }
    case TestPhase::FOLLOW_PATH_STAGE2: {
        if (!has_gps) {
            ROS_WARN_THROTTLE(1.0, "FOLLOW_PATH_STAGE2等待GPS初始化。");
            break;
        }
        bool finished = false;
        stage2_follow_step(vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd, finished);
        if (finished) {
            enter_phase(TestPhase::FINISH, now);
        }
        break;
    }
    case TestPhase::WAIT_STAGE2_MANUAL:
        vx_cmd = 0.0;
        vy_cmd = 0.0;
        vz_cmd = 0.0;
        yaw_rate_cmd = 0.0;
        if (!stage2_manual_zero_sent_) {
            stage2_manual_zero_sent_ = true;
        } else {
            publish_cmd = false;
        }
        ROS_INFO_THROTTLE(1.0, "WAIT_STAGE2_MANUAL: 自动段已结束，等待键控节点接管 /airsim_node/drone_1/vel_body_cmd");
        break;
    case TestPhase::FINAL_HOVER_TURN: {
        vx_cmd = 0.0;
        vy_cmd = 0.0;
        vz_cmd = 0.0;
        const double yaw_now_deg = quat_to_yaw_rad(current_quat) * 180.0 / M_PI;
        if (!final_turn_inited_) {
            final_turn_target_yaw_deg_ = wrap_deg(yaw_now_deg + final_turn_deg_);
            final_turn_inited_ = true;
            ROS_INFO("终点前悬停转向: 当前yaw=%.1f deg, 目标yaw=%.1f deg", yaw_now_deg, final_turn_target_yaw_deg_);
        }

        const double yaw_err_deg = wrap_deg(final_turn_target_yaw_deg_ - yaw_now_deg);
        if (std::fabs(yaw_err_deg) <= final_turn_tolerance_deg_) {
            yaw_rate_cmd = 0.0;
            if (enable_stage2_ && !stage2_active_) {
                if (stage2_manual_takeover_enable_) {
                    enter_phase(TestPhase::WAIT_STAGE2_MANUAL, now);
                } else if (load_stage2_path_from_anchor(current_pos)) {
                    stage2_hold_z_ = current_pos.z();
                    enter_phase(TestPhase::FOLLOW_PATH_STAGE2, now);
                } else {
                    ROS_ERROR("Stage2 path handoff failed, entering FINISH.");
                    enter_phase(TestPhase::FINISH, now);
                }
            } else {
                enter_phase(TestPhase::FINISH, now);
            }
        } else {
            yaw_rate_cmd = clampd(fp_kp_yaw_ * yaw_err_deg,
                                  -final_turn_rate_limit_deg_,
                                  final_turn_rate_limit_deg_);
            if (std::fabs(yaw_rate_cmd) < yaw_min_cmd_deg_) {
                yaw_rate_cmd = (yaw_err_deg >= 0.0) ? yaw_min_cmd_deg_ : -yaw_min_cmd_deg_;
            }
        }
        ROS_INFO_THROTTLE(0.5, "[FINAL_TURN] yaw_now=%.1f target=%.1f err=%.1f cmd=%.1f",
                          yaw_now_deg, final_turn_target_yaw_deg_, yaw_err_deg, yaw_rate_cmd);
        break;
    }
    case TestPhase::MOVE_FORWARD:
    case TestPhase::MOVE_BACKWARD:
    case TestPhase::MOVE_RIGHT:
    case TestPhase::MOVE_LEFT:
    case TestPhase::YAW_CW:
    case TestPhase::YAW_CCW:
        // 历史测试阶段保留占位，不再使用
        break;
    case TestPhase::FINISH:
        break;
    default:
        break;
    }

    if (publish_cmd) {
        publish_velocity_cmd(vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd, 0);
    }
    // [PERF] 出口计算本帧耗时，输出帧率指标
    const double perf_cb_ms = (ros::WallTime::now() - perf_loop_start).toSec() * 1000.0;
    perf_cb_ms_filt = (perf_cb_ms_filt <= 0.0) ? perf_cb_ms : (0.9 * perf_cb_ms_filt + 0.1 * perf_cb_ms);
    ROS_INFO_THROTTLE(0.5,
        "[PERF] cb_ms=%.2f cb_ms_avg=%.2f interval_ms=%.2f fps_avg=%.2f",
        perf_cb_ms, perf_cb_ms_filt, perf_interval_ms, perf_fps_filt);
    ROS_INFO_THROTTLE(0.5,
        "phase=%s cmd(vx,vy,vz,yawRate)=(%.2f,%.2f,%.2f,%.2f) pos=(%.2f,%.2f,%.2f) quat=(%.4f,%.4f,%.4f,%.4f)",
        phase_name(phase), vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd,
        current_pos.x(), current_pos.y(), current_pos.z(),
        current_quat.x(), current_quat.y(), current_quat.z(), current_quat.w());
}

void BasicDev::pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    const Eigen::Quaterniond q_raw(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
    if (pose_gt_is_ned_) {
        // pose_gt按NED解释：z向下为正，因此内部z-up时需要z取反并转换四元数
        current_quat = quat_ned_to_zup(q_raw);
        current_pos = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, -msg->pose.position.z);
    } else {
        // pose_gt已是内部坐标定义，直接使用
        current_quat = q_raw.normalized();
        current_pos = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }
    has_gps = true;  // 复用原状态变量，不改后续控制逻辑

    const Eigen::Vector3d eulerAngle = current_quat.toRotationMatrix().eulerAngles(2, 1, 0);
    constexpr double kPi = 3.14159265358979323846;
    ROS_INFO_THROTTLE(0.2,
        "POSE_GT(%s) t=%.3f pos=(%.3f, %.3f, %.3f) quat=(%.5f, %.5f, %.5f, %.5f) yaw=%.1f deg",
        pose_gt_is_ned_ ? "NED->ZUP" : "DIRECT",
        msg->header.stamp.sec + msg->header.stamp.nsec*1e-9,
        current_pos.x(), current_pos.y(), current_pos.z(),
        current_quat.x(), current_quat.y(), current_quat.z(), current_quat.w(),
        eulerAngle[0] * 180.0 / kPi);
}

void BasicDev::gps_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    const Eigen::Quaterniond q_ned(msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z);
    const Eigen::Quaterniond q = quat_ned_to_zup(q_ned);
    const Eigen::Vector3d eulerAngle = q.toRotationMatrix().eulerAngles(2, 1, 0);
    const Eigen::Vector3d p(msg->pose.position.x, msg->pose.position.y, -msg->pose.position.z);
    // 将控制状态切换为 GPS（NED->ZUP）
    current_quat = q;
    current_pos = p;
    has_gps = true;
    constexpr double kPi = 3.14159265358979323846;
    ROS_INFO_THROTTLE(0.2,
        "GPS(raw) t=%.3f pos=(%.3f, %.3f, %.3f) quat=(%.5f, %.5f, %.5f, %.5f) yaw=%.1f deg",
        msg->header.stamp.sec + msg->header.stamp.nsec*1e-9,
        p.x(), p.y(), p.z(),
        q.x(), q.y(), q.z(), q.w(),
        eulerAngle[0] * 180.0 / kPi);
}

void BasicDev::imu_cb(const sensor_msgs::Imu::ConstPtr& msg)
{
    ROS_INFO_THROTTLE(1.0, "Get imu data. time: %f", msg->header.stamp.sec + msg->header.stamp.nsec*1e-9);
}

void BasicDev::front_left_view_cb(const sensor_msgs::ImageConstPtr& msg)
{
    cv_front_left_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC3);
    if(!cv_front_left_ptr->image.empty())
    {
        ROS_INFO_THROTTLE(1.0, "Get front left image.: %f", msg->header.stamp.sec + msg->header.stamp.nsec*1e-9);
    }
}

void BasicDev::front_right_view_cb(const sensor_msgs::ImageConstPtr& msg)
{
    cv_front_right_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC3);
    if(!cv_front_right_ptr->image.empty())
    {
        ROS_INFO_THROTTLE(1.0, "Get front right image.%f", msg->header.stamp.sec + msg->header.stamp.nsec*1e-9);
    }
}

void BasicDev::back_left_view_cb(const sensor_msgs::ImageConstPtr& msg)
{
    cv_back_left_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC3);
    if(!cv_back_left_ptr->image.empty())
    {
        ROS_INFO_THROTTLE(1.0, "Get back left image.: %f", msg->header.stamp.sec + msg->header.stamp.nsec*1e-9);
    }
}

void BasicDev::back_right_view_cb(const sensor_msgs::ImageConstPtr& msg)
{
    cv_back_right_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC3);
    if(!cv_back_right_ptr->image.empty())
    {
        ROS_INFO_THROTTLE(1.0, "Get back right image.: %f", msg->header.stamp.sec + msg->header.stamp.nsec*1e-9);
    }
}

void BasicDev::lidar_cb(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr pts(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *pts);
    ROS_INFO_THROTTLE(1.0, "Get lidar data. time: %f, size: %ld", msg->header.stamp.sec + msg->header.stamp.nsec*1e-9, pts->size());

    if (!has_gps || !stage2_frame_cloud_body_) {
        return;
    }

    const Eigen::Quaterniond q_wb = current_quat.normalized();
    const Eigen::Vector3d t_w = current_pos;
    const double leaf = std::max(1e-3, corridor_voxel_leaf_);
    const double min_r = std::max(0.0, corridor_min_range_);
    const double max_r = std::max(min_r + 0.1, corridor_max_range_);
    const int scale_factor = std::max(1, stage2_process_scale_factor_);
    const int proc_yaw_bins = std::max(5, scale_factor * stage2_occupancy_grid_yaw_bins_);
    const int proc_pitch_bins = std::max(3, scale_factor * stage2_occupancy_grid_pitch_bins_);
    const double proc_half_yaw = clampd(stage2_occupancy_grid_fov_deg_ * M_PI / 180.0 * 0.5, 0.35, 1.45);
    const double proc_half_pitch = clampd(stage2_occupancy_grid_vfov_deg_ * M_PI / 180.0 * 0.5, 0.20, 1.20);
    const double yaw_span = 2.0 * proc_half_yaw;
    const double pitch_span = 2.0 * proc_half_pitch;
    const int min_points_per_cell = std::max(1, stage2_process_min_points_per_cell_);
    const ros::Time now = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
    const double elapsed_s =
        stage2_lidar_prev_stamp_.isZero() ? 0.0 : clampd((now - stage2_lidar_prev_stamp_).toSec(), 0.0, 0.20);
    stage2_lidar_prev_stamp_ = now;
    std::unordered_set<long long> frame_voxel_keys;
    frame_voxel_keys.reserve(pts->points.size());
    pcl::PointCloud<pcl::PointXYZI> frame_cloud_body;
    pcl::PointCloud<pcl::PointXYZ> frame_cloud;
    const size_t proc_cells = static_cast<size_t>(proc_yaw_bins * proc_pitch_bins);
    std::vector<int> cell_counts(proc_cells, 0);
    std::vector<double> cell_best_range(proc_cells, std::numeric_limits<double>::infinity());
    std::vector<pcl::PointXYZI> cell_best_points(proc_cells);
    std::vector<bool> cell_best_valid(proc_cells, false);
    frame_cloud_body.points.reserve(pts->points.size());
    frame_cloud.points.reserve(pts->points.size());

    for (const auto& p : pts->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;

        // Lidar原始点默认z-down，内部统一z-up
        const Eigen::Vector3d p_b(p.x, p.y, -p.z);
        const double r = p_b.norm();
        if (r < min_r || r > max_r) continue;

        const int ibx = static_cast<int>(std::floor(p_b.x() / leaf));
        const int iby = static_cast<int>(std::floor(p_b.y() / leaf));
        const int ibz = static_cast<int>(std::floor(p_b.z() / leaf));
        const long long body_key = pack_voxel_key(ibx, iby, ibz);
        if (frame_voxel_keys.find(body_key) != frame_voxel_keys.end()) continue;
        frame_voxel_keys.insert(body_key);
        const double bearing = std::atan2(p_b.y(), std::max(1e-3, p_b.x()));
        const double elev = std::atan2(p_b.z(), std::max(1e-3, p_b.head<2>().norm()));
        if (bearing < -proc_half_yaw || bearing > proc_half_yaw) continue;
        if (elev < -proc_half_pitch || elev > proc_half_pitch) continue;
        const double yaw_ratio = (bearing + proc_half_yaw) / std::max(1e-3, yaw_span);
        const double pitch_ratio = (elev + proc_half_pitch) / std::max(1e-3, pitch_span);
        const int yaw_bin = std::max(0, std::min(proc_yaw_bins - 1,
            static_cast<int>(std::floor(yaw_ratio * static_cast<double>(proc_yaw_bins)))));
        const int pitch_bin = std::max(0, std::min(proc_pitch_bins - 1,
            static_cast<int>(std::floor(pitch_ratio * static_cast<double>(proc_pitch_bins)))));
        const size_t cell = static_cast<size_t>(pitch_bin * proc_yaw_bins + yaw_bin);
        cell_counts[cell] += 1;
        if (!cell_best_valid[cell] || r < cell_best_range[cell]) {
            pcl::PointXYZI body_pt;
            body_pt.x = static_cast<float>(p_b.x());
            body_pt.y = static_cast<float>(p_b.y());
            body_pt.z = static_cast<float>(p_b.z());
            body_pt.intensity = 0.0f;
            cell_best_points[cell] = body_pt;
            cell_best_range[cell] = r;
            cell_best_valid[cell] = true;
        }
        const Eigen::Vector3d p_w = t_w + q_wb * p_b;
        frame_cloud.points.emplace_back(static_cast<float>(p_w.x()),
                                        static_cast<float>(p_w.y()),
                                        static_cast<float>(p_w.z()));
    }

    for (size_t cell = 0; cell < proc_cells; ++cell) {
        if (cell_counts[cell] >= min_points_per_cell && cell_best_valid[cell]) {
            frame_cloud_body.points.push_back(cell_best_points[cell]);
        }
    }

    if (stage2_frame_cloud_body_ && !stage2_frame_cloud_body_->points.empty()) {
        for (const auto& old_p : stage2_frame_cloud_body_->points) {
            const Eigen::Vector3d p_b(old_p.x, old_p.y, old_p.z);
            const double r = p_b.norm();
            if (r < min_r || r > max_r) continue;
            if (old_p.intensity + elapsed_s > stage2_process_max_age_sec_) continue;
            const double bearing = std::atan2(p_b.y(), std::max(1e-3, p_b.x()));
            const double elev = std::atan2(p_b.z(), std::max(1e-3, p_b.head<2>().norm()));
            if (bearing < -proc_half_yaw || bearing > proc_half_yaw) continue;
            if (elev < -proc_half_pitch || elev > proc_half_pitch) continue;
            const double yaw_ratio = (bearing + proc_half_yaw) / std::max(1e-3, yaw_span);
            const double pitch_ratio = (elev + proc_half_pitch) / std::max(1e-3, pitch_span);
            const int yaw_bin = std::max(0, std::min(proc_yaw_bins - 1,
                static_cast<int>(std::floor(yaw_ratio * static_cast<double>(proc_yaw_bins)))));
            const int pitch_bin = std::max(0, std::min(proc_pitch_bins - 1,
                static_cast<int>(std::floor(pitch_ratio * static_cast<double>(proc_pitch_bins)))));
            const size_t cell = static_cast<size_t>(pitch_bin * proc_yaw_bins + yaw_bin);
            if (cell_counts[cell] >= min_points_per_cell) continue;
            pcl::PointXYZI keep = old_p;
            keep.intensity = static_cast<float>(old_p.intensity + elapsed_s);
            frame_cloud_body.points.push_back(keep);
            cell_counts[cell] = min_points_per_cell;
        }
    }

    frame_cloud_body.width = static_cast<uint32_t>(frame_cloud_body.points.size());
    frame_cloud_body.height = 1;
    *stage2_frame_cloud_body_ = frame_cloud_body;

    sensor_msgs::PointCloud2 frame_msg;
    pcl::toROSMsg(frame_cloud, frame_msg);
    frame_msg.header.stamp = now;
    frame_msg.header.frame_id = viz_frame_id_;
    lidar_frame_pub_.publish(frame_msg);

    if (corridor_enable_) {
        const double min_pub_period = 1.0 / std::max(0.1, corridor_pub_hz_);
        if (last_corridor_pub_time_.isZero() || (now - last_corridor_pub_time_).toSec() >= min_pub_period) {
            sensor_msgs::PointCloud2 map_msg;
            pcl::toROSMsg(frame_cloud, map_msg);
            map_msg.header.stamp = now;
            map_msg.header.frame_id = viz_frame_id_;
            corridor_map_pub_.publish(map_msg);
            last_corridor_pub_time_ = now;
            ROS_INFO_THROTTLE(1.0, "Current lidar frame points: %zu", frame_cloud.points.size());
        }
    }
}

void BasicDev::pwm_feedback_cb(const airsim_ros::RotorPWM::ConstPtr& msg)
{
    ROS_INFO_THROTTLE(1.0, "Get rotor pwm feedback. %f %f %f %f", msg->rotorPWM0, msg->rotorPWM1, msg->rotorPWM2, msg->rotorPWM3);
}

void BasicDev::initial_pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    init_pos = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, -msg->pose.position.z);
    has_init_pose = true;
    ROS_INFO("Get initial pose. time: %f, posi: %f, %f, %f", msg->header.stamp.sec + msg->header.stamp.nsec*1e-9,
        msg->pose.position.x, msg->pose.position.y, -msg->pose.position.z);
}

void BasicDev::end_goal_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    // 内部统一 z-up 坐标
    goal_pos = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y, -msg->pose.position.z);
    ROS_INFO("Get end goal. time: %f, goal_pos: %f, %f, %f",
        msg->header.stamp.sec + msg->header.stamp.nsec*1e-9,
        goal_pos.x(), goal_pos.y(), goal_pos.z());
    ROS_INFO("goal_orientation: x=%f y=%f z=%f w=%f",
        msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w);
}

#endif
