#!/usr/bin/env python3

import csv
import json
import math
import os
import re
from collections import deque
from datetime import datetime

import rospy
from airsim_ros.msg import VelCmd
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Log
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PREVIEW_RE = re.compile(
    rf"\\[PX4_PREVIEW\\] ok=(?P<ok>\\d+) lidar=(?P<lidar>\\d+) "
    rf"goal=\\((?P<goal_x>{FLOAT_RE}) (?P<goal_y>{FLOAT_RE}) (?P<goal_z>{FLOAT_RE})\\) "
    rf"wp=\\((?P<wp_x>{FLOAT_RE}) (?P<wp_y>{FLOAT_RE}) (?P<wp_z>{FLOAT_RE})\\) "
    rf"vel=\\((?P<vel_x>{FLOAT_RE}) (?P<vel_y>{FLOAT_RE}) (?P<vel_z>{FLOAT_RE})\\)"
)
PREVIEW_WAIT_RE = re.compile(
    r"\[PX4_PREVIEW\] waiting: lidar=(?P<lidar>\d+) path_loaded=(?P<path_loaded>\d+) planner=(?P<planner>\d+)"
)
PATH_RE = re.compile(
    rf"\\[PATH\\] near=(?P<near>\\d+) tar=(?P<tar>\\d+) tar_z=(?P<tar_z>\\d+) "
    rf"L=(?P<L>{FLOAT_RE}) vh=(?P<vh>{FLOAT_RE}) yaw_err=(?P<yaw_err>{FLOAT_RE}) "
    rf"wccw=(?P<wccw>{FLOAT_RE}) dz=(?P<dz>{FLOAT_RE}) vref=(?P<vref>{FLOAT_RE}) "
    rf"vz=(?P<vz>{FLOAT_RE}) zi=(?P<zi>{FLOAT_RE}) vd=(?P<vd>{FLOAT_RE}) "
    rf"dist_end=(?P<dist_end>{FLOAT_RE}) e_b=\\((?P<eb_x>{FLOAT_RE}) (?P<eb_y>{FLOAT_RE}) (?P<eb_z>{FLOAT_RE})\\) "
    rf"cmd=\\((?P<cmd_vx>{FLOAT_RE}) (?P<cmd_vy>{FLOAT_RE}) (?P<cmd_vz>{FLOAT_RE}) (?P<cmd_yaw>{FLOAT_RE})\\)"
)
PHASE_RE = re.compile(
    rf"phase=(?P<phase>\\S+) cmd\\(vx,vy,vz,yawRate\\)=\\((?P<cmd_vx>{FLOAT_RE}),(?P<cmd_vy>{FLOAT_RE}),(?P<cmd_vz>{FLOAT_RE}),(?P<cmd_yaw>{FLOAT_RE})\\) "
    rf"pos=\\((?P<pos_x>{FLOAT_RE}),(?P<pos_y>{FLOAT_RE}),(?P<pos_z>{FLOAT_RE})\\) "
    rf"quat=\\((?P<qx>{FLOAT_RE}),(?P<qy>{FLOAT_RE}),(?P<qz>{FLOAT_RE}),(?P<qw>{FLOAT_RE})\\)"
)
ENTER_PHASE_RE = re.compile(r"进入阶段: (?P<phase>\S+)")


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def wrap_pi(rad):
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def quat_to_yaw_deg(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def quat_ned_to_zup(qx, qy, qz, qw):
    # Reflection with S = diag(1, 1, -1), same convention as basic_dev.
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    r11 = 1.0 - 2.0 * (yy + zz)
    r12 = 2.0 * (xy - wz)
    r13 = 2.0 * (xz + wy)
    r21 = 2.0 * (xy + wz)
    r22 = 1.0 - 2.0 * (xx + zz)
    r23 = 2.0 * (yz - wx)
    r31 = 2.0 * (xz - wy)
    r32 = 2.0 * (yz + wx)
    r33 = 1.0 - 2.0 * (xx + yy)

    r13 = -r13
    r23 = -r23
    r31 = -r31
    r32 = -r32

    trace = r11 + r22 + r33
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw2 = 0.25 * s
        qx2 = (r32 - r23) / s
        qy2 = (r13 - r31) / s
        qz2 = (r21 - r12) / s
    elif r11 > r22 and r11 > r33:
        s = math.sqrt(1.0 + r11 - r22 - r33) * 2.0
        qw2 = (r32 - r23) / s
        qx2 = 0.25 * s
        qy2 = (r12 + r21) / s
        qz2 = (r13 + r31) / s
    elif r22 > r33:
        s = math.sqrt(1.0 + r22 - r11 - r33) * 2.0
        qw2 = (r13 - r31) / s
        qx2 = (r12 + r21) / s
        qy2 = 0.25 * s
        qz2 = (r23 + r32) / s
    else:
        s = math.sqrt(1.0 + r33 - r11 - r22) * 2.0
        qw2 = (r21 - r12) / s
        qx2 = (r13 + r31) / s
        qy2 = (r23 + r32) / s
        qz2 = 0.25 * s

    norm = math.sqrt(qx2 * qx2 + qy2 * qy2 + qz2 * qz2 + qw2 * qw2)
    if norm < 1e-9:
        return 0.0, 0.0, 0.0, 1.0
    return qx2 / norm, qy2 / norm, qz2 / norm, qw2 / norm


def to_float_map(match):
    data = {}
    for key, value in match.groupdict().items():
        if value in ("0", "1") and key in ("ok", "lidar", "path_loaded", "planner", "near", "tar", "tar_z"):
            data[key] = int(value)
        else:
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
    return data


class FlightSessionRecorder:
    def __init__(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_dir = os.path.join("/tmp", f"basic_dev_flight_{ts}")
        self.output_dir = rospy.get_param("~output_dir", default_dir)
        self.sample_hz = rospy.get_param("~sample_hz", 20.0)
        self.duration_sec = rospy.get_param("~duration_sec", 0.0)
        self.pose_gt_is_ned = rospy.get_param("~pose_gt_is_ned", True)

        os.makedirs(self.output_dir, exist_ok=True)

        self.telemetry_file = open(os.path.join(self.output_dir, "telemetry.csv"), "w", newline="")
        self.preview_file = open(os.path.join(self.output_dir, "preview_events.csv"), "w", newline="")
        self.preview_wait_file = open(os.path.join(self.output_dir, "preview_waiting.csv"), "w", newline="")
        self.path_file = open(os.path.join(self.output_dir, "path_events.csv"), "w", newline="")
        self.phase_file = open(os.path.join(self.output_dir, "phase_events.csv"), "w", newline="")
        self.phase_enter_file = open(os.path.join(self.output_dir, "phase_enter.csv"), "w", newline="")
        self.rosout_file = open(os.path.join(self.output_dir, "rosout_filtered.log"), "w")

        self.telemetry_writer = csv.writer(self.telemetry_file)
        self.preview_writer = csv.writer(self.preview_file)
        self.preview_wait_writer = csv.writer(self.preview_wait_file)
        self.path_writer = csv.writer(self.path_file)
        self.phase_writer = csv.writer(self.phase_file)
        self.phase_enter_writer = csv.writer(self.phase_enter_file)

        self.telemetry_writer.writerow([
            "stamp",
            "pos_x", "pos_y", "pos_z", "yaw_deg",
            "vel_x", "vel_y", "vel_z", "speed_xy",
            "cmd_vx", "cmd_vy", "cmd_vz", "cmd_yaw_rate",
            "target_x", "target_y", "target_z",
            "lidar_points",
            "preview_ok", "preview_lidar",
            "preview_goal_x", "preview_goal_y", "preview_goal_z",
            "preview_wp_x", "preview_wp_y", "preview_wp_z",
            "preview_vel_x", "preview_vel_y", "preview_vel_z",
            "path_near", "path_tar", "path_L", "path_yaw_err", "path_dist_end",
            "path_cmd_vx", "path_cmd_vy", "path_cmd_vz", "path_cmd_yaw",
            "phase_name", "phase_cmd_vx", "phase_cmd_vy", "phase_cmd_vz", "phase_cmd_yaw",
        ])
        self.preview_writer.writerow([
            "stamp", "ok", "lidar",
            "goal_x", "goal_y", "goal_z",
            "wp_x", "wp_y", "wp_z",
            "vel_x", "vel_y", "vel_z",
        ])
        self.preview_wait_writer.writerow(["stamp", "lidar", "path_loaded", "planner"])
        self.path_writer.writerow([
            "stamp", "near", "tar", "tar_z", "L", "vh", "yaw_err", "wccw", "dz", "vref", "vz", "zi", "vd",
            "dist_end", "eb_x", "eb_y", "eb_z", "cmd_vx", "cmd_vy", "cmd_vz", "cmd_yaw",
        ])
        self.phase_writer.writerow([
            "stamp", "phase", "cmd_vx", "cmd_vy", "cmd_vz", "cmd_yaw",
            "pos_x", "pos_y", "pos_z", "qx", "qy", "qz", "qw",
        ])
        self.phase_enter_writer.writerow(["stamp", "phase"])

        self.pose = None
        self.cmd = None
        self.target = None
        self.lidar_points = 0
        self.latest_preview = {}
        self.latest_path = {}
        self.latest_phase = {}
        self.latest_enter_phase = ""

        self.vel_est = [0.0, 0.0, 0.0]
        self.prev_pose_for_vel = None
        self.prev_pose_stamp = None
        self.vel_alpha = clamp(rospy.get_param("~vel_est_alpha", 0.3), 0.0, 1.0)

        self.event_counts = {
            "preview": 0,
            "preview_wait": 0,
            "path": 0,
            "phase": 0,
            "phase_enter": 0,
        }
        self._recent_log_keys = set()
        self._recent_log_queue = deque()

        rospy.Subscriber("/airsim_node/drone_1/debug/pose_gt", PoseStamped, self.pose_cb, queue_size=50)
        rospy.Subscriber("/airsim_node/drone_1/vel_body_cmd", VelCmd, self.cmd_cb, queue_size=50)
        rospy.Subscriber("/viz/target_point", Marker, self.target_cb, queue_size=10)
        rospy.Subscriber("/airsim_node/drone_1/lidar", PointCloud2, self.lidar_cb, queue_size=5)
        rospy.Subscriber("/rosout", Log, self.rosout_cb, queue_size=200)
        rospy.Subscriber("/rosout_agg", Log, self.rosout_cb, queue_size=200)

        self._write_meta()
        rospy.loginfo("Flight session recorder output: %s", self.output_dir)

    def _write_meta(self):
        meta = {
            "created_at": datetime.now().isoformat(),
            "output_dir": self.output_dir,
            "sample_hz": self.sample_hz,
            "duration_sec": self.duration_sec,
            "pose_gt_is_ned": self.pose_gt_is_ned,
            "files": {
                "telemetry": "telemetry.csv",
                "preview_events": "preview_events.csv",
                "preview_waiting": "preview_waiting.csv",
                "path_events": "path_events.csv",
                "phase_events": "phase_events.csv",
                "phase_enter": "phase_enter.csv",
                "rosout_filtered": "rosout_filtered.log",
            },
        }
        with open(os.path.join(self.output_dir, "session_meta.json"), "w") as meta_file:
            json.dump(meta, meta_file, indent=2, sort_keys=True)

    def pose_cb(self, msg):
        q = msg.pose.orientation
        if self.pose_gt_is_ned:
            qx, qy, qz, qw = quat_ned_to_zup(q.x, q.y, q.z, q.w)
            pos = (msg.pose.position.x, msg.pose.position.y, -msg.pose.position.z)
        else:
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            pos = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

        stamp = msg.header.stamp.to_sec() if not msg.header.stamp.is_zero() else rospy.Time.now().to_sec()
        if self.prev_pose_for_vel is not None and self.prev_pose_stamp is not None:
            dt = clamp(stamp - self.prev_pose_stamp, 0.005, 0.20)
            raw_vel = [
                (pos[0] - self.prev_pose_for_vel[0]) / max(1e-3, dt),
                (pos[1] - self.prev_pose_for_vel[1]) / max(1e-3, dt),
                (pos[2] - self.prev_pose_for_vel[2]) / max(1e-3, dt),
            ]
            self.vel_est = [
                (1.0 - self.vel_alpha) * self.vel_est[0] + self.vel_alpha * raw_vel[0],
                (1.0 - self.vel_alpha) * self.vel_est[1] + self.vel_alpha * raw_vel[1],
                (1.0 - self.vel_alpha) * self.vel_est[2] + self.vel_alpha * raw_vel[2],
            ]

        self.prev_pose_for_vel = pos
        self.prev_pose_stamp = stamp
        self.pose = {
            "stamp": stamp,
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "yaw_deg": quat_to_yaw_deg(qx, qy, qz, qw),
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
        }

    def cmd_cb(self, msg):
        self.cmd = {
            "vx": msg.vx,
            "vy": msg.vy,
            "vz": msg.vz,
            "yaw_rate": msg.yawRate,
        }

    def target_cb(self, msg):
        p = msg.pose.position
        self.target = {"x": p.x, "y": p.y, "z": p.z}

    def lidar_cb(self, msg):
        self.lidar_points = int(msg.width * msg.height)

    def rosout_cb(self, msg):
        text = msg.msg
        stamp = msg.header.stamp.to_sec() if not msg.header.stamp.is_zero() else rospy.Time.now().to_sec()
        key = (msg.header.stamp.secs, msg.header.stamp.nsecs, msg.name, text)
        if key in self._recent_log_keys:
            return
        self._recent_log_keys.add(key)
        self._recent_log_queue.append(key)
        while len(self._recent_log_queue) > 4000:
            old = self._recent_log_queue.popleft()
            self._recent_log_keys.discard(old)

        interesting = any(tag in text for tag in ("[PX4_PREVIEW]", "[PATH]", "phase=", "进入阶段:"))
        if interesting:
            self.rosout_file.write(f"{stamp:.6f} {msg.name} {text}\n")

        preview_match = PREVIEW_RE.search(text)
        if preview_match:
            data = to_float_map(preview_match)
            self.latest_preview = data
            self.preview_writer.writerow([
                f"{stamp:.6f}", data["ok"], data["lidar"],
                f"{data['goal_x']:.6f}", f"{data['goal_y']:.6f}", f"{data['goal_z']:.6f}",
                f"{data['wp_x']:.6f}", f"{data['wp_y']:.6f}", f"{data['wp_z']:.6f}",
                f"{data['vel_x']:.6f}", f"{data['vel_y']:.6f}", f"{data['vel_z']:.6f}",
            ])
            self.event_counts["preview"] += 1
            return

        preview_wait_match = PREVIEW_WAIT_RE.search(text)
        if preview_wait_match:
            data = {k: int(v) for k, v in preview_wait_match.groupdict().items()}
            self.preview_wait_writer.writerow([f"{stamp:.6f}", data["lidar"], data["path_loaded"], data["planner"]])
            self.event_counts["preview_wait"] += 1
            return

        path_match = PATH_RE.search(text)
        if path_match:
            data = to_float_map(path_match)
            self.latest_path = data
            self.path_writer.writerow([
                f"{stamp:.6f}",
                int(data["near"]), int(data["tar"]), int(data["tar_z"]),
                f"{data['L']:.6f}", f"{data['vh']:.6f}", f"{data['yaw_err']:.6f}", f"{data['wccw']:.6f}",
                f"{data['dz']:.6f}", f"{data['vref']:.6f}", f"{data['vz']:.6f}", f"{data['zi']:.6f}",
                f"{data['vd']:.6f}", f"{data['dist_end']:.6f}",
                f"{data['eb_x']:.6f}", f"{data['eb_y']:.6f}", f"{data['eb_z']:.6f}",
                f"{data['cmd_vx']:.6f}", f"{data['cmd_vy']:.6f}", f"{data['cmd_vz']:.6f}", f"{data['cmd_yaw']:.6f}",
            ])
            self.event_counts["path"] += 1
            return

        phase_match = PHASE_RE.search(text)
        if phase_match:
            data = to_float_map(phase_match)
            self.latest_phase = data
            self.phase_writer.writerow([
                f"{stamp:.6f}", data["phase"],
                f"{data['cmd_vx']:.6f}", f"{data['cmd_vy']:.6f}", f"{data['cmd_vz']:.6f}", f"{data['cmd_yaw']:.6f}",
                f"{data['pos_x']:.6f}", f"{data['pos_y']:.6f}", f"{data['pos_z']:.6f}",
                f"{data['qx']:.6f}", f"{data['qy']:.6f}", f"{data['qz']:.6f}", f"{data['qw']:.6f}",
            ])
            self.event_counts["phase"] += 1
            return

        enter_phase_match = ENTER_PHASE_RE.search(text)
        if enter_phase_match:
            self.latest_enter_phase = enter_phase_match.group("phase")
            self.phase_enter_writer.writerow([f"{stamp:.6f}", self.latest_enter_phase])
            self.event_counts["phase_enter"] += 1

    def _sample_row(self):
        if self.pose is None:
            return

        cmd = self.cmd or {}
        target = self.target or {}
        preview = self.latest_preview or {}
        path = self.latest_path or {}
        phase = self.latest_phase or {}

        speed_xy = math.hypot(self.vel_est[0], self.vel_est[1])
        self.telemetry_writer.writerow([
            f"{self.pose['stamp']:.6f}",
            f"{self.pose['x']:.6f}", f"{self.pose['y']:.6f}", f"{self.pose['z']:.6f}", f"{self.pose['yaw_deg']:.6f}",
            f"{self.vel_est[0]:.6f}", f"{self.vel_est[1]:.6f}", f"{self.vel_est[2]:.6f}", f"{speed_xy:.6f}",
            f"{cmd.get('vx', float('nan')):.6f}", f"{cmd.get('vy', float('nan')):.6f}",
            f"{cmd.get('vz', float('nan')):.6f}", f"{cmd.get('yaw_rate', float('nan')):.6f}",
            f"{target.get('x', float('nan')):.6f}", f"{target.get('y', float('nan')):.6f}", f"{target.get('z', float('nan')):.6f}",
            self.lidar_points,
            int(preview.get("ok", -1)), int(preview.get("lidar", -1)),
            f"{preview.get('goal_x', float('nan')):.6f}", f"{preview.get('goal_y', float('nan')):.6f}",
            f"{preview.get('goal_z', float('nan')):.6f}",
            f"{preview.get('wp_x', float('nan')):.6f}", f"{preview.get('wp_y', float('nan')):.6f}",
            f"{preview.get('wp_z', float('nan')):.6f}",
            f"{preview.get('vel_x', float('nan')):.6f}", f"{preview.get('vel_y', float('nan')):.6f}",
            f"{preview.get('vel_z', float('nan')):.6f}",
            int(path.get("near", -1)), int(path.get("tar", -1)),
            f"{path.get('L', float('nan')):.6f}", f"{path.get('yaw_err', float('nan')):.6f}",
            f"{path.get('dist_end', float('nan')):.6f}",
            f"{path.get('cmd_vx', float('nan')):.6f}", f"{path.get('cmd_vy', float('nan')):.6f}",
            f"{path.get('cmd_vz', float('nan')):.6f}", f"{path.get('cmd_yaw', float('nan')):.6f}",
            phase.get("phase", self.latest_enter_phase),
            f"{phase.get('cmd_vx', float('nan')):.6f}", f"{phase.get('cmd_vy', float('nan')):.6f}",
            f"{phase.get('cmd_vz', float('nan')):.6f}", f"{phase.get('cmd_yaw', float('nan')):.6f}",
        ])

        self.telemetry_file.flush()
        self.preview_file.flush()
        self.preview_wait_file.flush()
        self.path_file.flush()
        self.phase_file.flush()
        self.phase_enter_file.flush()
        self.rosout_file.flush()

    def close(self):
        summary = {
            "event_counts": self.event_counts,
            "closed_at": datetime.now().isoformat(),
        }
        with open(os.path.join(self.output_dir, "summary.json"), "w") as summary_file:
            json.dump(summary, summary_file, indent=2, sort_keys=True)

        for handle in (
            self.telemetry_file,
            self.preview_file,
            self.preview_wait_file,
            self.path_file,
            self.phase_file,
            self.phase_enter_file,
            self.rosout_file,
        ):
            handle.close()

    def run(self):
        rate = rospy.Rate(max(1.0, self.sample_hz))
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            self._sample_row()
            if self.duration_sec > 0.0 and (rospy.Time.now().to_sec() - start_time) >= self.duration_sec:
                break
            rate.sleep()

        self.close()
        rospy.loginfo("Flight session recorder saved logs to %s", self.output_dir)


if __name__ == "__main__":
    rospy.init_node("flight_session_recorder", anonymous=False)
    recorder = FlightSessionRecorder()
    try:
        recorder.run()
    finally:
        if not recorder.telemetry_file.closed:
            recorder.close()
