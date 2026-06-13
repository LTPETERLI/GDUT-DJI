#include "kb_ctrl.hpp"

#include <X11/Xutil.h>

namespace {
constexpr size_t KEY_YAW_LEFT = 0;
constexpr size_t KEY_FORWARD = 1;
constexpr size_t KEY_YAW_RIGHT = 2;
constexpr size_t KEY_DOWN = 3;
constexpr size_t KEY_LEFT = 4;
constexpr size_t KEY_BACKWARD = 5;
constexpr size_t KEY_RIGHT = 6;
constexpr size_t KEY_UP = 7;
constexpr size_t KEY_CLEAR = 8;
constexpr size_t KEY_QUIT = 9;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kb_ctrl");
    ros::NodeHandle nh("~");
    KBCtrl ctrl(&nh);
    ros::spin();
    return 0;
}

KBCtrl::KBCtrl(ros::NodeHandle* nh) : nh_(nh)
{
    nh_->param("forward_speed", forward_speed_, 8.0);
    nh_->param("backward_speed", backward_speed_, 3.0);
    nh_->param("lateral_speed", lateral_speed_, 5.0);
    nh_->param("vertical_speed", vertical_speed_, 4.0);
    nh_->param("yaw_rate_deg", yaw_rate_deg_, 35.0);

    velcmd_.vx = 0.0;
    velcmd_.vy = 0.0;
    velcmd_.vz = 0.0;
    velcmd_.yawRate = 0.0;
    velcmd_.va = 10.0;
    velcmd_.stop = 0;

    vel_publisher_ = nh_->advertise<airsim_ros::VelCmd>("/airsim_node/drone_1/vel_body_cmd", 1);
    takeoff_client_ = nh_->serviceClient<airsim_ros::Takeoff>("/airsim_node/drone_1/takeoff");
    land_client_ = nh_->serviceClient<airsim_ros::Land>("/airsim_node/drone_1/land");
    reset_client_ = nh_->serviceClient<airsim_ros::Reset>("/airsim_node/reset");

    ROS_INFO("kb_ctrl ready. keys: w/s x, a/d y, f/space z, j/l yaw, c clear, t takeoff, p land, r reset, Esc quit");

    kb_thread_ = std::thread(&KBCtrl::keyboard_loop, this);
    timer_ = nh_->createTimer(ros::Duration(0.02), &KBCtrl::ctrl_callback, this);
}

KBCtrl::~KBCtrl()
{
    if (display_) {
        XAutoRepeatOn(display_);
        XUngrabKeyboard(display_, CurrentTime);
        XCloseDisplay(display_);
        display_ = nullptr;
    }
    if (kb_thread_.joinable()) kb_thread_.join();
}

void KBCtrl::keyboard_loop()
{
    display_ = XOpenDisplay(nullptr);
    if (!display_) {
        ROS_ERROR("kb_ctrl: unable to open X display");
        return;
    }

    root_ = DefaultRootWindow(display_);
    XGrabKeyboard(display_, root_, True, GrabModeAsync, GrabModeAsync, CurrentTime);
    XSelectInput(display_, root_, KeyPressMask | KeyReleaseMask);
    XAutoRepeatOff(display_);

    while (ros::ok()) {
        XNextEvent(display_, &event_);
        char buffer[16] = {0};
        KeySym keysym;
        XLookupString(&event_.xkey, buffer, sizeof(buffer), &keysym, nullptr);

        const bool pressed = (event_.type == KeyPress);
        if (event_.type != KeyPress && event_.type != KeyRelease) continue;

        if (keysym == XK_Escape) {
            key_status_[KEY_QUIT] = pressed;
            ros::shutdown();
            break;
        }

        switch (buffer[0]) {
        case 'j': key_status_[KEY_YAW_LEFT] = pressed; break;
        case 'w': key_status_[KEY_FORWARD] = pressed; break;
        case 'l': key_status_[KEY_YAW_RIGHT] = pressed; break;
        case ' ': key_status_[KEY_DOWN] = pressed; break;
        case 'a': key_status_[KEY_LEFT] = pressed; break;
        case 's': key_status_[KEY_BACKWARD] = pressed; break;
        case 'd': key_status_[KEY_RIGHT] = pressed; break;
        case 'f': key_status_[KEY_UP] = pressed; break;
        case 'c': key_status_[KEY_CLEAR] = pressed; break;
        case 't':
            if (pressed) takeoff_client_.call(takeoff_);
            break;
        case 'p':
            if (pressed) land_client_.call(land_);
            break;
        case 'r':
            if (pressed) reset_client_.call(reset_);
            break;
        default:
            break;
        }
    }
}

void KBCtrl::ctrl_callback(const ros::TimerEvent&)
{
    velcmd_.header.stamp = ros::Time::now();
    velcmd_.vx = 0.0;
    velcmd_.vy = 0.0;
    velcmd_.vz = 0.0;
    velcmd_.yawRate = 0.0;
    velcmd_.stop = 0;

    if (key_status_[KEY_FORWARD]) velcmd_.vx = forward_speed_;
    if (key_status_[KEY_BACKWARD]) velcmd_.vx = -backward_speed_;
    if (key_status_[KEY_LEFT]) velcmd_.vy = -lateral_speed_;
    if (key_status_[KEY_RIGHT]) velcmd_.vy = lateral_speed_;
    if (key_status_[KEY_UP]) velcmd_.vz = vertical_speed_;
    if (key_status_[KEY_DOWN]) velcmd_.vz = -vertical_speed_;
    if (key_status_[KEY_YAW_LEFT]) velcmd_.yawRate = -yaw_rate_deg_;
    if (key_status_[KEY_YAW_RIGHT]) velcmd_.yawRate = yaw_rate_deg_;

    if (key_status_[KEY_CLEAR]) {
        velcmd_.vx = 0.0;
        velcmd_.vy = 0.0;
        velcmd_.vz = 0.0;
        velcmd_.yawRate = 0.0;
    }

    vel_publisher_.publish(velcmd_);
}
