#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
航点录制器 (record_waypoints.py)

订阅 /airsim_node/drone_1/debug/pose_gt,按距离间隔采样,录成与现有CSV航迹
同格式的文件 (t,x,y,z,qx,qy,qz,qw),可直接当自动轨迹复用。

用途: 手飞交接后录制人工飞的轨迹航点。
配合 kb_ctrl 键盘手飞使用。

用法:
  python3 record_waypoints.py                       # 默认输出/tmp,间隔0.5m
  python3 record_waypoints.py --out my.csv --dist 0.3
  # Ctrl+C 停止并保存

注意: 直接录 pose_gt 原始值(NED,z向下),与现有 gnss_path*.csv 一致,
       加载时 basic_dev 会按 pose_gt_is_ned 自动转换。
"""
import argparse
import math
import signal
import sys
from datetime import datetime

import rospy
from geometry_msgs.msg import PoseStamped


class WaypointRecorder:
    def __init__(self, out_path, min_dist):
        self.out_path = out_path
        self.min_dist = min_dist
        self.rows = []          # (t,x,y,z,qx,qy,qz,qw)
        self.last_xyz = None
        self.sub = rospy.Subscriber(
            "/airsim_node/drone_1/debug/pose_gt", PoseStamped,
            self.pose_cb, queue_size=50)
        rospy.loginfo("航点录制中... 按距离间隔 %.2fm 采样, Ctrl+C 停止保存到 %s",
                      self.min_dist, self.out_path)

    def pose_cb(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        t = msg.header.stamp.to_sec()
        if self.last_xyz is not None:
            d = math.sqrt((p.x - self.last_xyz[0])**2 +
                          (p.y - self.last_xyz[1])**2 +
                          (p.z - self.last_xyz[2])**2)
            if d < self.min_dist:
                return
        self.rows.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        self.last_xyz = (p.x, p.y, p.z)
        if len(self.rows) % 20 == 0:
            rospy.loginfo("已录 %d 个航点", len(self.rows))

    def save(self):
        if not self.rows:
            rospy.logwarn("无航点可保存(没收到pose_gt?)")
            return
        with open(self.out_path, "w") as f:
            f.write("# t,x,y,z,qx,qy,qz,qw\n")
            for r in self.rows:
                f.write("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" % r)
        rospy.loginfo("已保存 %d 个航点 → %s", len(self.rows), self.out_path)


def main():
    ap = argparse.ArgumentParser()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap.add_argument("--out", default=f"/tmp/manual_waypoints_{ts}.csv")
    ap.add_argument("--dist", type=float, default=0.5, help="采样距离间隔(m)")
    args, _ = ap.parse_known_args()

    rospy.init_node("record_waypoints", anonymous=True)
    rec = WaypointRecorder(args.out, args.dist)

    def on_shutdown():
        rec.save()
    rospy.on_shutdown(on_shutdown)

    rospy.spin()


if __name__ == "__main__":
    main()
