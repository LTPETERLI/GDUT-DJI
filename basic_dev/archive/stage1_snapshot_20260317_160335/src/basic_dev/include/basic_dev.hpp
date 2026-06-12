#ifndef _BASIC_DEV_HPP_
#define _BASIC_DEV_HPP_

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include "airsim_ros/VelCmd.h"
#include "airsim_ros/RotorPWM.h"
#include "airsim_ros/PoseCmd.h"
#include "airsim_ros/Takeoff.h"
#include "airsim_ros/Reset.h"
#include "airsim_ros/Land.h"
#include "airsim_ros/GPSYaw.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseStamped.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Imu.h"
#include "std_msgs/Float64.h"
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <time.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include <ros/callback_queue.h>
#include <boost/thread/thread.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_set>
#include <limits>

class BasicDev
{
private:
    ros::NodeHandle* nh_ = nullptr;
    // ------------------------------
    // 传感器缓存
    // ------------------------------
    cv_bridge::CvImageConstPtr cv_front_left_ptr, cv_front_right_ptr, cv_back_left_ptr, cv_back_right_ptr;
    cv::Mat front_left_img, front_right_img, back_left_img, back_right_img;

    std::unique_ptr<image_transport::ImageTransport> it;
    ros::CallbackQueue go_queue;
    ros::CallbackQueue front_img_queue;

    // ------------------------------
    // 服务请求与控制消息
    // ------------------------------
    airsim_ros::Takeoff takeoff;
    airsim_ros::Land land;
    airsim_ros::Reset reset;
    airsim_ros::VelCmd velcmd;

    // ------------------------------
    // ROS 通信接口
    // ------------------------------
    ros::Subscriber odom_suber; // 状态真值
    ros::Subscriber gps_suber; // gps数据
    ros::Subscriber imu_suber; // imu数据
    ros::Subscriber lidar_suber; // lidar数据
    ros::Subscriber pwm_feedback_suber; // 电机输入PWM信号
    ros::Subscriber initial_pose_suber; // 起始位姿
    ros::Subscriber end_goal_suber; // 终点位置
    image_transport::Subscriber front_left_view_suber;
    image_transport::Subscriber front_right_view_suber;
    image_transport::Subscriber back_left_view_suber;
    image_transport::Subscriber back_right_view_suber;

    // 通过这两个服务可以调用模拟器中的无人机起飞和降落命令
    ros::ServiceClient takeoff_client;
    ros::ServiceClient land_client;
    ros::ServiceClient reset_client;

    // 通过publisher实现对无人机的速度控制
    ros::Publisher vel_publisher;
    ros::Publisher desired_path_pub_;
    ros::Publisher actual_path_pub_;
    ros::Publisher target_marker_pub_;
    ros::Publisher corridor_map_pub_;
    ros::Publisher lidar_frame_pub_;
    ros::Publisher stage2_obstacle_points_pub_;
    ros::Publisher stage2_debug_marker_pub_;
    ros::Publisher dbg_z_sat_pub_;
    ros::Publisher dbg_z_dz_pub_;
    ros::Publisher dbg_z_tar_pub_;
    ros::Publisher dbg_z_tar_xy_pub_;
    ros::Publisher dbg_z_cur_pub_;
    ros::Publisher dbg_z_vref_pub_;
    ros::Publisher dbg_z_vz_meas_pub_;
    ros::Publisher dbg_z_vz_cmd_pub_;
    ros::Timer control_timer;

    // ------------------------------
    // 任务状态
    // ------------------------------
    enum class TestPhase {
        WAITING_DATA,
        HOVER_AFTER_TAKEOFF,
        MOVE_FORWARD_BUFFER,
        HOVER_AFTER_BUFFER,
        FOLLOW_PATH,
        FINAL_HOVER_TURN,
        WAIT_STAGE2_MANUAL,
        FOLLOW_PATH_STAGE2,
        MOVE_FORWARD,
        MOVE_BACKWARD,
        MOVE_RIGHT,
        MOVE_LEFT,
        YAW_CW,
        YAW_CCW,
        FINISH
    };
    enum class Stage2FollowMode {
        CRUISE,
        AVOID
    };

    struct Stage2OccupancyGrid {
        int yaw_bins = 0;
        int pitch_bins = 0;
        double yaw_min = 0.0;
        double yaw_max = 0.0;
        double pitch_min = 0.0;
        double pitch_max = 0.0;
        double max_range = 0.0;
        std::vector<int> point_counts;
        std::vector<double> min_ranges;
        std::vector<double> mean_ranges;
        std::vector<double> cell_costs;
        int total_points = 0;
    };

    bool has_gps = false;
    bool has_init_pose = false;
    bool has_takeoff = false;
    bool use_takeoff_service_ = false; // false=不调用起飞服务，直接进入控制流程
    bool path_loaded_ = false;
    TestPhase phase = TestPhase::WAITING_DATA;
    ros::Time last_takeoff_try;
    ros::Time phase_start_time;
    ros::Time takeoff_success_time;

    Eigen::Vector3d current_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d init_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d goal_pos = Eigen::Vector3d::Zero();
    Eigen::Quaterniond current_quat = Eigen::Quaterniond::Identity();

    // ------------------------------
    // 测试动作参数
    // ------------------------------
    double test_speed_xy = 10.0; // m/s
    double test_speed_z = 0.0; // 本序列不做升降
    double test_yaw_rate_deg = 45.0; // deg/s（接口常用角速度单位）
    double test_move_dist = 2.0; // m
    double test_hover_sec = 0.0; // s
    double buffer_dist_ = 10.0; // m，起飞后前飞缓冲距离
    double buffer_vx_ = 16.0; // m/s，机体系前向速度
    double buffer_vz_ = 0.8; // m/s，缓冲段机体系竖直速度（正值上升）
    double buffer_max_climb_ = 0.9; // m，缓冲段相对起点最大爬升，超过后停止继续上抬
    double buffer_hover_sec_ = 0.0; // s，缓冲后悬停时间
    double max_vx_ = 18.0;
    double max_vy_ = 12.0;
    double max_vz_ = 12.0;
    double max_vz_start_zone_ = 2.6; // 起点缓冲后前段Z速度上限（m/s）

    // CSV航迹跟踪参数
    std::string csv_path_;
    std::string stage2_csv_path_;
    std::vector<Eigen::Vector3d> path_pts_;
    std::vector<double> path_s_; // 轨迹累计弧长，用于按进度切换控制参数
    double downsample_dist_ = 0.8;
    double lookahead_dist_ = 8.0;
    double lookahead_min_dist_ = 6.0; // 自适应前视的最小距离（m）
    double lookahead_max_dist_ = 10.0; // 自适应前视的最大距离（m）
    double lookahead_time_gain_ = 0.25; // 自适应前视的速度增益（m / (m/s)）
    double lookahead_speed_ref_ = 10.0; // 超过该水平速度后再逐步放大前视（m/s）
    double reach_end_dist_ = 0.6;
    double final_brake_start_dist_ = 16.0; // 终点前开始提前刹车的距离（m）
    double final_brake_min_scale_ = 0.20; // 提前刹车时保留的最小速度比例
    double final_stop_before_end_dist_ = 8.0; // 终点前提前停下的距离（m）
    double final_turn_deg_ = 180.0; // 终点前悬停后原地转向角度（deg）
    double final_turn_rate_limit_deg_ = 35.0; // 终点原地转向最大角速度（deg/s）
    double final_turn_tolerance_deg_ = 2.0; // 终点原地转向完成判定（deg）
    bool enable_stage2_ = true;
    bool stage2_invert_csv_y_ = false;
    bool use_relative_path_ = false;
    bool invert_csv_y_ = true;
    double fp_kp_xy_ = 1.8;
    double fp_kp_x_ = 1.8; // 机体系X轴位置误差增益
    double fp_kp_y_ = 4.2; // 机体系Y轴位置误差增益
    double fp_y_vel_damp_ = 1.0; // 机体系Y轴速度阻尼增益（越大越稳）
    double fp_vy_lpf_alpha_ = 0.20; // vy命令低通滤波系数(0~1)
    double lateral_slow_err_thresh_ = 4.0; // |e_b.y|超过该阈值才触发前向降速（m）
    double late_lateral_err_thresh_ = 8.0; // 飞行到后段时放宽的横向门控阈值（m）
    double gate_relax_time_sec_ = 90.0;    // 放宽门控阈值的时间点（秒，起飞后）
    double lateral_slow_gain_ = 0.22; // 超阈值后的前向降速强度
    double lateral_slow_min_scale_ = 0.70; // 前向降速最小比例，避免速度掉太多
    double z_lookahead_dist_ = 0.3; // Z目标前视距离（m），小前视更贴合轨迹
    double max_vz_down_ = 10.0; // 下拉最大速度（m/s）
    // 传统单环Z轴PID参数
    double z_pid_kp_ = 5.2;
    double z_pid_ki_ = 1.4;
    double z_pid_kd_ = 0.8;
    double z_pid_i_limit_ = 2.2; // 积分限幅（m*s）
    double z_pid_d_lpf_alpha_ = 0.35; // 速度误差低通滤波系数(0~1)
    double z_vel_lpf_alpha_ = 0.30; // 实际竖直速度低通滤波系数(0~1)
    double z_ref_vel_lpf_alpha_ = 0.35; // 目标竖直速度低通滤波系数(0~1)
    double z_ref_vel_ff_gain_ = 1.6; // 目标竖直速度前馈增益
    double upward_gate_dy_thresh_ = 3.0; // |dy|超过该值时限制上升速度
    double upward_gate_vz_cap_ = 2.0;    // 侧偏大时的上升速度上限（m/s）
    // Z轴PID状态
    double z_pid_int_ = 0.0;
    double z_pid_prev_err_ = 0.0;
    double z_pid_d_filt_ = 0.0; // 当前使用速度误差滤波值 v_err_filt
    bool z_pid_inited_ = false;
    ros::Time z_pid_prev_time_;
    // Z轴速度环状态（用于D项）
    bool z_vel_inited_ = false;
    double z_prev_meas_ = 0.0;
    double z_prev_ref_ = 0.0;
    double z_vel_meas_filt_ = 0.0;
    double z_ref_vel_filt_ = 0.0;
    // XY速度估计与Y通道平滑状态
    bool xy_vel_inited_ = false;
    bool xy_speed_inited_ = false;
    Eigen::Vector3d prev_pos_for_xy_vel_ = Eigen::Vector3d::Zero();
    ros::Time prev_xy_vel_time_;
    double vy_cmd_filt_ = 0.0;
    double fp_kp_z_turn_boost_ = 1.35; // 转弯爬升时的Z轴增强增益
    double max_vz_turn_boost_ = 3.2; // 转弯爬升时的Z速度上限（m/s）
    double z_turn_boost_yaw_deg_ = 22.0; // 触发转弯增强的最小偏航误差（deg）
    double z_turn_boost_dz_ = 0.25; // 触发转弯增强的最小爬升需求（m）
    double fp_kp_yaw_ = 3.0;
    double fp_kd_yaw_rate_ = 0.22; // yaw D增益（对实测偏航角速度反馈，单位deg/s）
    double yaw_rate_lpf_alpha_ = 0.32; // 偏航角速度低通滤波系数(0~1)
    double yaw_soft_limit_deg_ = 120.0; // yawRate平滑限幅尺度（deg/s）
    double yaw_stop_tolerance_deg_ = 0.8; // 到点停止容差（deg）
    double yaw_min_cmd_deg_ = 3.5; // 停止容差外的最小yawRate，避免末段过慢（deg/s）
    bool enable_yaw_only_gate_ = true; // 是否启用大角误差“只转不走”门控
    double yaw_only_deg_ = 65.0;
    bool yaw_rate_inited_ = false;
    double yaw_prev_rad_ = 0.0;
    double yaw_rate_meas_filt_ = 0.0; // 实测偏航角速度（CCW为正，deg/s）
    ros::Time yaw_prev_time_;
    bool final_turn_inited_ = false;
    double final_turn_target_yaw_deg_ = 0.0;
    // -------------------------------------------------------------------------
    // Stage2 planner / controller state
    // -------------------------------------------------------------------------
    bool stage2_active_ = false;
    bool stage2_manual_takeover_enable_ = false;
    bool stage2_manual_zero_sent_ = false;
    Stage2FollowMode stage2_mode_ = Stage2FollowMode::CRUISE;
    Eigen::Vector3d stage2_start_pos_ = Eigen::Vector3d::Zero();
    bool stage2_start_pos_valid_ = false;
    double stage2_hold_z_ = 0.0;
    double stage2_z_blend_pause_xy_ = 0.0;
    double stage2_prev_travel_xy_ = 0.0;
    bool stage2_avoid_enable_ = true;
    double stage2_max_vx_ = 8.0;
    double stage2_max_vy_ = 4.0;
    double stage2_max_vz_ = 8.0;
    double stage2_avoid_max_vy_ = 12.0;
    double stage2_avoid_max_vz_ = 10.0;
    double stage2_avoid_vy_gain_scale_ = 1.8;
    double stage2_avoid_vz_gain_scale_ = 1.4;
    double stage2_avoid_min_vy_cmd_ = 3.5;
    double stage2_recover_max_vz_ = 2.0;
    double stage2_avoid_vy_slew_rate_ = 12.0;
    double stage2_avoid_vz_slew_rate_ = 10.0;
    double stage2_avoid_max_pitch_deg_ = 18.0;
    double stage2_avoid_vertical_weight_ = 1.1;
    double stage2_avoid_vertical_rejoin_weight_ = 0.8;
    double stage2_avoid_candidate_switch_margin_ = 3.0;
    double stage2_avoid_target_xy_slew_rate_ = 6.0;
    double stage2_avoid_target_z_slew_rate_ = 2.5;
    int stage2_z_avoid_trigger_frames_ = 3;
    double stage2_dynamic_avoid_max_vy_ = 12.0;
    double stage2_dynamic_avoid_max_vz_ = 12.0;
    double stage2_dynamic_avoid_vy_gain_scale_ = 2.4;
    double stage2_dynamic_avoid_vz_gain_scale_ = 1.9;
    double stage2_dynamic_avoid_vy_slew_rate_ = 18.0;
    double stage2_dynamic_avoid_vz_slew_rate_ = 16.0;
    double stage2_dynamic_avoid_max_lateral_offset_ = 1.2;
    double stage2_z_blend_dist_ = 4.0;
    double stage2_avoid_forward_range_ = 12.0;
    double stage2_avoid_lateral_half_width_ = 1.8;
    double stage2_avoid_vertical_half_height_ = 1.6;
    double stage2_avoid_center_half_width_ = 0.9;
    double stage2_avoid_side_probe_width_ = 2.5;
    double stage2_avoid_max_lateral_offset_ = 2.0;
    int stage2_avoid_candidate_count_ = 7;
    double stage2_avoid_ray_radius_ = 0.9;
    int stage2_blockage_min_points_ = 6;
    double stage2_blockage_fov_deg_ = 60.0;
    double stage2_blockage_vertical_fov_deg_ = 45.0;
    double stage2_blockage_density_thresh_ = 0.18;
    int stage2_blockage_sector_bins_ = 9;
    int stage2_blockage_vertical_bins_ = 5;
    double stage2_avoid_clearance_weight_ = 1.6;
    double stage2_avoid_alignment_weight_ = 0.9;
    double stage2_avoid_offset_weight_ = 0.7;
    double stage2_avoid_density_weight_ = 0.6;
    int stage2_avoid_density_sector_bins_ = 5;
    int stage2_avoid_density_vertical_bins_ = 3;
    double stage2_gap_safe_width_ = 0.15;
    double stage2_gap_width_weight_ = 1.2;
    double stage2_corridor_width_ = 10.0;
    double stage2_corridor_height_ = 5.0;
    double stage2_vehicle_width_ = 0.40;
    double stage2_vehicle_height_ = 0.30;
    double stage2_vehicle_clearance_ = 0.15;
    double stage2_vehicle_safe_width_ = 0.70;
    double stage2_vehicle_safe_height_ = 0.60;
    double stage2_corridor_half_width_safe_ = 4.65;
    double stage2_corridor_half_height_safe_ = 2.20;
    double stage2_occupancy_grid_fov_deg_ = 140.0;
    double stage2_occupancy_grid_vfov_deg_ = 70.0;
    int stage2_occupancy_grid_yaw_bins_ = 19;
    int stage2_occupancy_grid_pitch_bins_ = 7;
    int stage2_process_scale_factor_ = 3;
    int stage2_process_min_points_per_cell_ = 1;
    double stage2_process_max_age_sec_ = 0.30;
    double stage2_avoid_trigger_dist_ = 1.5;
    double stage2_avoid_slowdown_dist_ = 8.0;
    double stage2_avoid_stop_dist_ = 4.0;
    double stage2_avoid_min_scale_ = 0.15;
    double stage2_recover_clearance_ = 5.0;
    double stage2_recover_hold_sec_ = 0.6;
    double stage2_dynamic_closing_speed_thresh_ = 1.2;
    double stage2_dynamic_ttc_thresh_ = 1.0;
    double stage2_local_traj_spacing_ = 0.8;
    double stage2_local_traj_lookahead_dist_ = 2.0;
    // Debug and telemetry exported to RViz / rosout for second-stage diagnosis.
    std::vector<Eigen::Vector3d> stage2_debug_candidate_targets_;
    std::vector<double> stage2_debug_candidate_scores_;
    std::vector<double> stage2_debug_candidate_clearances_;
    std::vector<double> stage2_debug_candidate_gap_widths_;
    int stage2_debug_selected_candidate_ = -1;
    double stage2_debug_blockage_dist_ = std::numeric_limits<double>::infinity();
    int stage2_debug_blockage_points_ = 0;
    double stage2_debug_blockage_density_ = 0.0;
    int stage2_debug_blockage_occupied_bins_ = 0;
    double stage2_debug_best_candidate_clearance_ = std::numeric_limits<double>::infinity();
    double stage2_debug_best_gap_width_ = 0.0;
    // [METRIC] 候选可行率埋点：可行候选数 / 候选窗口总数；fallback 标志
    int stage2_debug_feasible_candidates_ = 0;
    int stage2_debug_total_candidates_ = 0;
    bool stage2_debug_fallback_used_ = false;
    bool stage2_cmd_inited_ = false;
    double stage2_prev_vy_cmd_ = 0.0;
    double stage2_prev_vz_cmd_ = 0.0;
    bool stage2_avoid_latched_ = false;
    int stage2_triggered_streak_ = 0;
    Eigen::Vector3d stage2_last_avoid_target_ = Eigen::Vector3d::Zero();
    bool stage2_last_avoid_target_valid_ = false;
    Eigen::Vector3d stage2_smoothed_avoid_target_ = Eigen::Vector3d::Zero();
    bool stage2_smoothed_avoid_target_valid_ = false;
    int stage2_last_candidate_yaw_idx_ = -1;
    int stage2_last_candidate_pitch_idx_ = -1;
    std::vector<Eigen::Vector3d> stage2_local_traj_;
    ros::Time stage2_last_blocked_time_;
    double stage2_prev_blockage_dist_ = std::numeric_limits<double>::infinity();
    ros::Time stage2_prev_blockage_time_;
    // -------------------------------------------------------------------------
    // Stage1 turn-guard and generic controller fallback parameters
    // -------------------------------------------------------------------------
    bool stage1_turn_guard_enable_ = true;
    double stage1_turn_guard_s_min_ = 900.0;
    double stage1_turn_guard_s_max_ = 980.0;
    double stage1_turn_guard_lookahead_ = 8.0;
    double stage1_turn_guard_max_vx_ = 13.0;
    double stage1_turn_guard_max_vy_ = 4.0;
    bool use_pure_p_control_ = false; // 置true时，XY/Z/Yaw 只做P计算，不做限幅、慢区、只转不走等附加逻辑
    // pose_gt坐标系开关：true=按NED输入并转到内部z-up，false=直接按z-up使用
    bool pose_gt_is_ned_ = true;

    // -------------------------------------------------------------------------
    // RViz / visualization buffers
    // -------------------------------------------------------------------------
    nav_msgs::Path desired_path_msg_;
    nav_msgs::Path actual_path_msg_;
    bool rviz_inited_ = false;
    double actual_append_dist_ = 0.05;
    size_t actual_max_points_ = 3000;
    Eigen::Vector3d last_actual_pos_ = Eigen::Vector3d::Zero();
    bool last_actual_pos_valid_ = false;
    std::string viz_frame_id_ = "map";
    bool publish_follow_tf_ = true;
    std::string follow_frame_id_ = "drone_follow";
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    // 走廊可视化（把激光点累积到世界坐标）
    bool corridor_enable_ = true;
    double corridor_voxel_leaf_ = 0.25;
    double corridor_min_range_ = 0.6;
    double corridor_max_range_ = 45.0;
    double corridor_pub_hz_ = 2.0;
    size_t corridor_max_points_ = 250000;
    ros::Time last_corridor_pub_time_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr corridor_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr stage2_frame_cloud_body_;
    ros::Time stage2_lidar_prev_stamp_;
    std::unordered_set<long long> corridor_voxel_keys_;

    double clamp_value(double value, double lower, double upper);
    const char* phase_name(TestPhase p) const;
    const char* stage2_mode_name(Stage2FollowMode m) const;
    void enter_phase(TestPhase p, const ros::Time& now);
    double phase_elapsed(const ros::Time& now) const;
    bool load_path_from_csv(const Eigen::Vector3d& start_world);
    bool load_stage2_path_from_anchor(const Eigen::Vector3d& anchor_world);
    size_t find_nearest_index(const Eigen::Vector3d& p) const;
    size_t find_lookahead_index(size_t i_near, double L) const;
    bool build_stage2_occupancy_grid(const Eigen::Vector3d& raw_target,
                                     Stage2OccupancyGrid& grid) const;
    bool compute_stage2_blockage_metrics(const Stage2OccupancyGrid& grid,
                                         double& nearest_along,
                                         int& hit_count,
                                         double& density,
                                         int& occupied_bins) const;
    bool compute_stage2_avoid_target(const Eigen::Vector3d& raw_target,
                                     const Stage2OccupancyGrid& grid,
                                     Eigen::Vector3d& avoid_target,
                                     double& avoid_offset);
    void refresh_stage2_geometry_baseline();
    void publish_stage2_obstacle_points() const;
    void publish_stage2_debug_markers(const Eigen::Vector3d& raw_target,
                                      const Eigen::Vector3d& avoid_target,
                                      double obs_dist,
                                      double avoid_offset,
                                      double avoid_scale) const;
    void path_follow_step(double& vx, double& vy, double& vz, double& yaw_deg, bool& finished);
    void stage2_follow_step(double& vx, double& vy, double& vz, double& yaw_deg, bool& finished);
    void publish_desired_path_once();
    void update_and_publish_actual_path();
    void publish_target_marker(const Eigen::Vector3d& p_tar);
    void publish_follow_tf();
    void publish_velocity_cmd(double vx, double vy, double vz, double yaw_rate, uint8_t stop);
    void control_timer_cb(const ros::TimerEvent& event);

    void pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void gps_cb(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void imu_cb(const sensor_msgs::Imu::ConstPtr& msg);
    void lidar_cb(const sensor_msgs::PointCloud2::ConstPtr& msg);

    void front_left_view_cb(const sensor_msgs::ImageConstPtr& msg);
    void front_right_view_cb(const sensor_msgs::ImageConstPtr& msg);
    void back_left_view_cb(const sensor_msgs::ImageConstPtr& msg);
    void back_right_view_cb(const sensor_msgs::ImageConstPtr& msg);
    void pwm_feedback_cb(const airsim_ros::RotorPWM::ConstPtr& msg);
    void initial_pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void end_goal_cb(const geometry_msgs::PoseStamped::ConstPtr& msg);

public:
    BasicDev(ros::NodeHandle* nh);
    ~BasicDev();
};

#endif
