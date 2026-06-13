#ifndef _KB_CTRL_HPP_
#define _KB_CTRL_HPP_

#include <array>
#include <thread>

#include <X11/Xlib.h>

#include <ros/ros.h>

#include "airsim_ros/Land.h"
#include "airsim_ros/Reset.h"
#include "airsim_ros/Takeoff.h"
#include "airsim_ros/VelCmd.h"

class KBCtrl {
public:
    explicit KBCtrl(ros::NodeHandle* nh);
    ~KBCtrl();

private:
    ros::NodeHandle* nh_ = nullptr;
    ros::Publisher vel_publisher_;
    ros::ServiceClient takeoff_client_;
    ros::ServiceClient land_client_;
    ros::ServiceClient reset_client_;
    ros::Timer timer_;

    airsim_ros::VelCmd velcmd_;
    airsim_ros::Takeoff takeoff_;
    airsim_ros::Land land_;
    airsim_ros::Reset reset_;

    Display* display_ = nullptr;
    Window root_;
    XEvent event_;
    std::thread kb_thread_;
    std::array<bool, 10> key_status_{};

    double forward_speed_ = 8.0;
    double backward_speed_ = 3.0;
    double lateral_speed_ = 5.0;
    double vertical_speed_ = 4.0;
    double yaw_rate_deg_ = 35.0;

    void keyboard_loop();
    void ctrl_callback(const ros::TimerEvent&);
};

#endif
