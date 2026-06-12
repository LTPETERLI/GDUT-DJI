#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单飞行日志记录器：
- 默认记录 140s
- 输出 CSV 到 /tmp/rmua_flight_log_<timestamp>.csv
- 订阅:
  /airsim_node/drone_1/debug/pose_gt   (geometry_msgs/PoseStamped)
  /airsim_node/drone_1/vel_body_cmd    (airsim_ros/VelCmd)
  /viz/target_point                    (visualization_msgs/Marker)
CSV字段:
  stamp, pos_x, pos_y, pos_z, yaw_deg,
  cmd_vx, cmd_vy, cmd_vz, cmd_yawRate,
  tgt_x, tgt_y, tgt_z
"""

import csv
import math
import time
from datetime import datetime

import rospy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from airsim_ros.msg import VelCmd


def quat_to_yaw_deg(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


class FlightLogger:
    def __init__(self):
        self.duration = rospy.get_param("~log_duration_sec", 140.0)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"/tmp/rmua_flight_log_{ts}.csv"
        self.log_path = rospy.get_param("~log_path", default_path)

        self.pose = None
        self.cmd = None
        self.target = None

        rospy.Subscriber("/airsim_node/drone_1/debug/pose_gt", PoseStamped, self.pose_cb, queue_size=20)
        rospy.Subscriber("/airsim_node/drone_1/vel_body_cmd", VelCmd, self.cmd_cb, queue_size=20)
        rospy.Subscriber("/viz/target_point", Marker, self.target_cb, queue_size=10)

        self.fh = open(self.log_path, "w", newline="")
        self.writer = csv.writer(self.fh)
        self.writer.writerow([
            "stamp",
            "pos_x", "pos_y", "pos_z", "yaw_deg",
            "cmd_vx", "cmd_vy", "cmd_vz", "cmd_yawRate",
            "tgt_x", "tgt_y", "tgt_z"
        ])
        rospy.loginfo("FlightLogger started: duration=%.1fs, output=%s", self.duration, self.log_path)

    def pose_cb(self, msg: PoseStamped):
        self.pose = msg

    def cmd_cb(self, msg: VelCmd):
        self.cmd = msg

    def target_cb(self, msg: Marker):
        self.target = msg

    def run(self):
        rate = rospy.Rate(30)  # 30Hz 够用
        t0 = time.time()
        while not rospy.is_shutdown():
            now = time.time()
            if now - t0 > self.duration:
                break
            if self.pose is None or self.cmd is None:
                rate.sleep()
                continue

            # 当前姿态
            p = self.pose.pose.position
            q = self.pose.pose.orientation
            yaw_deg = quat_to_yaw_deg(q)

            # 当前指令
            cmd = self.cmd

            # 目标点
            if self.target is not None:
                tp = self.target.pose.position
                tgt_x, tgt_y, tgt_z = tp.x, tp.y, tp.z
            else:
                tgt_x = tgt_y = tgt_z = float("nan")

            stamp = self.pose.header.stamp.to_sec()
            self.writer.writerow([
                f"{stamp:.6f}",
                f"{p.x:.6f}", f"{p.y:.6f}", f"{p.z:.6f}", f"{yaw_deg:.3f}",
                f"{cmd.vx:.3f}", f"{cmd.vy:.3f}", f"{cmd.vz:.3f}", f"{cmd.yawRate:.3f}",
                f"{tgt_x:.6f}", f"{tgt_y:.6f}", f"{tgt_z:.6f}"
            ])
            rate.sleep()

        self.fh.close()
        rospy.loginfo("FlightLogger done. Saved: %s", self.log_path)


if __name__ == "__main__":
    rospy.init_node("flight_logger", anonymous=False)
    FlightLogger().run()
