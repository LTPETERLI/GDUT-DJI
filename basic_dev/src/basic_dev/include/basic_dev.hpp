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

class BasicDev
{
private:
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
        MOVE_FORWARD,
        MOVE_BACKWARD,
        MOVE_RIGHT,
        MOVE_LEFT,
        YAW_CW,
        YAW_CCW,
        FINISH
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
    double buffer_vx_ = 14.0; // m/s，机体系前向速度
    double buffer_vz_ = 0.8; // m/s，缓冲段机体系竖直速度（正值上升）
    double buffer_max_climb_ = 0.9; // m，缓冲段相对起点最大爬升，超过后停止继续上抬
    double buffer_hover_sec_ = 0.0; // s，缓冲后悬停时间
    double max_vx_ = 14.0;
    double max_vy_ = 18.0;
    double max_vz_ = 4.5;
    double max_vz_start_zone_ = 2.6; // 起点缓冲后前段Z速度上限（m/s）

    // CSV航迹跟踪参数
    std::string csv_path_;
    std::vector<Eigen::Vector3d> path_pts_;
    std::vector<double> path_s_; // 轨迹累计弧长，用于按进度切换控制参数
    double downsample_dist_ = 0.8;
    double lookahead_dist_ = 7.5;
    double reach_end_dist_ = 0.6;
    bool use_relative_path_ = true;
    bool invert_csv_y_ = true;
    double fp_kp_xy_ = 1.5;
    double fp_kp_x_ = 2.2; // 机体系X轴位置误差增益
    double fp_kp_y_ = 5.2; // 机体系Y轴位置误差增益
    double fp_y_vel_damp_ = 1.2; // 机体系Y轴速度阻尼增益（越大越稳）
    double fp_vy_lpf_alpha_ = 0.25; // vy命令低通滤波系数(0~1)
    double lateral_slow_err_thresh_ = 6.0; // |e_b.y|超过该阈值才触发前向降速（m）
    double lateral_slow_gain_ = 0.22; // 超阈值后的前向降速强度
    double lateral_slow_min_scale_ = 0.70; // 前向降速最小比例，避免速度掉太多
    double z_lookahead_dist_ = 0.3; // Z目标前视距离（m），小前视更贴合轨迹
    double max_vz_down_ = 6.0; // 下拉最大速度（m/s）
    // 传统单环Z轴PID参数
    double z_pid_kp_ = 5.2;
    double z_pid_ki_ = 1.4;
    double z_pid_kd_ = 0.8;
    double z_pid_i_limit_ = 2.2; // 积分限幅（m*s）
    double z_pid_d_lpf_alpha_ = 0.35; // 速度误差低通滤波系数(0~1)
    double z_vel_lpf_alpha_ = 0.30; // 实际竖直速度低通滤波系数(0~1)
    double z_ref_vel_lpf_alpha_ = 0.35; // 目标竖直速度低通滤波系数(0~1)
    double z_ref_vel_ff_gain_ = 2.2; // 目标竖直速度前馈增益
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
    bool enable_yaw_only_gate_ = false; // 是否启用大角误差“只转不走”门控
    double yaw_only_deg_ = 65.0;
    bool yaw_rate_inited_ = false;
    double yaw_prev_rad_ = 0.0;
    double yaw_rate_meas_filt_ = 0.0; // 实测偏航角速度（CCW为正，deg/s）
    ros::Time yaw_prev_time_;
    bool use_pure_p_control_ = false; // 置true时，XY/Z/Yaw 只做P计算，不做限幅、慢区、只转不走等附加逻辑
    // pose_gt坐标系开关：true=按NED输入并转到内部z-up，false=直接按z-up使用
    bool pose_gt_is_ned_ = true;

    // RViz可视化缓存
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
    std::unordered_set<long long> corridor_voxel_keys_;

    double clamp_value(double value, double lower, double upper);
    const char* phase_name(TestPhase p) const;
    void enter_phase(TestPhase p, const ros::Time& now);
    double phase_elapsed(const ros::Time& now) const;
    bool load_path_from_csv(const Eigen::Vector3d& start_world);
    size_t find_nearest_index(const Eigen::Vector3d& p) const;
    size_t find_lookahead_index(size_t i_near, double L) const;
    void path_follow_step(double& vx, double& vy, double& vz, double& yaw_deg, bool& finished);
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
