#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray

from airsim_ros.msg import VelCmd


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_pi(rad):
    while rad > math.pi:
        rad -= 2.0 * math.pi
    while rad < -math.pi:
        rad += 2.0 * math.pi
    return rad


def quat_to_yaw(q):
    # 标准Z轴偏航角（CCW为正）
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class YawTuner(object):
    def __init__(self):
        rospy.init_node("yaw_tuner", anonymous=False)

        # 话题参数
        self.pose_topic = rospy.get_param("~pose_topic", "/airsim_node/drone_1/debug/pose_gt")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/airsim_node/drone_1/vel_body_cmd")
        self.target_topic = rospy.get_param("~target_topic", "/yaw_tuner/target_yaw_deg")
        self.marker_topic = rospy.get_param("~marker_topic", "/yaw_tuner/markers")
        self.cur_yaw_topic = rospy.get_param("~current_yaw_topic", "/yaw_tuner/current_yaw_deg")
        self.err_yaw_topic = rospy.get_param("~yaw_error_topic", "/yaw_tuner/yaw_error_deg")
        self.cmd_yaw_topic = rospy.get_param("~yaw_rate_cmd_topic", "/yaw_tuner/yaw_rate_cmd_deg")

        # 控制参数（默认值采用“调试友好”保守配置）
        # yaw P 调试模式：yaw_err(deg) * fp_kp_yaw（可选择不做限幅）
        self.use_basic_dev_yaw_logic = rospy.get_param("~use_basic_dev_yaw_logic", True)
        self.fp_kp_yaw = rospy.get_param("~fp_kp_yaw", 3.0)  # P增益：建议0.5~3.0，过大易振荡，过小响应慢
        self.fp_kd_yaw_rate = rospy.get_param("~fp_kd_yaw_rate", 0.26)  # D增益(角速度反馈)：建议0.0~0.6，过大易迟滞，过小抑振差
        self.pure_p_mode = rospy.get_param("~pure_p_mode", False)       # 仅用P项（不做限幅/斜坡），便于定位方向问题
        self.yaw_soft_limit_deg = rospy.get_param("~yaw_soft_limit_deg", 120.0)  # 输出软限幅(deg/s)：建议30~120，过大易过冲，过小转向慢
        self.yaw_stop_tolerance_deg = rospy.get_param("~yaw_stop_tolerance_deg", 0.8)  # 到点停止容差(deg)：建议0.5~5，过小易抖动，过大有静差
        self.yaw_min_cmd_deg = rospy.get_param("~yaw_min_cmd_deg", 6.0)  # 最小角速度(deg/s)：接近目标时防止爬行，建议2~8
        self.disable_yaw_limit = rospy.get_param("~disable_yaw_limit", False)
        self.disable_yaw_stop_tolerance = rospy.get_param("~disable_yaw_stop_tolerance", False)
        # 角速度斜坡限制（deg/s^2）：建议80~300，过大易突变，过小响应拖沓
        self.max_yaw_accel_deg_s2 = rospy.get_param("~max_yaw_accel_deg_s2", 300.0)
        self.yaw_rate_lpf_alpha = clamp(rospy.get_param("~yaw_rate_lpf_alpha", 0.32), 0.0, 1.0)  # 角速度低通系数：建议0.1~0.5，越小越平滑

        # PID 分支（仅在 use_basic_dev_yaw_logic=false 时使用）
        self.kp = rospy.get_param("~kp", 8.0)
        self.ki = rospy.get_param("~ki", 0.8)
        self.kd = rospy.get_param("~kd", 0.25)
        self.i_limit = rospy.get_param("~i_limit", 1.5)  # rad*s
        self.max_yaw_rate_deg = rospy.get_param("~max_yaw_rate_deg", 220.0)
        self.deadband_deg = rospy.get_param("~deadband_deg", 0.8)
        self.d_lpf_alpha = clamp(rospy.get_param("~d_lpf_alpha", 0.35), 0.0, 1.0)
        self.loop_hz = max(5.0, rospy.get_param("~loop_hz", 50.0))
        # 方向相关参数写死：避免被历史 rosparam 覆盖
        self.yaw_invert_direction = False
        # 统一约定：指令正值 = 顺时针。若仿真/传感坐标系与可视化相反，可通过 invert_pose_yaw 修正读取的姿态。
        self.cmd_cw_positive = True
        # 如果发现正指令却逆时针，请把 invert_pose_yaw 设为 false（或相反再设为 true）
        self.invert_pose_yaw = True
        self.require_target_input = rospy.get_param("~require_target_input", True)
        self.viz_frame_id = rospy.get_param("~viz_frame_id", "map")
        self.arrow_len = rospy.get_param("~arrow_len", 1.5)
        self.arrow_z_up = rospy.get_param("~arrow_z_up", 0.25)
        self.draw_on_plane = rospy.get_param("~draw_on_plane", True)
        self.draw_plane_z = rospy.get_param("~draw_plane_z", 0.3)
        self.show_trail = rospy.get_param("~show_trail", True)
        self.trail_max_points = max(20, rospy.get_param("~trail_max_points", 300))
        self.trail_min_step = max(0.001, rospy.get_param("~trail_min_step", 0.02))
        # 仅影响可视化：若 RViz 中箭头朝向与实机相反，可翻转或加偏置，不影响控制逻辑
        self.viz_invert_yaw = False
        self.viz_yaw_offset_rad = math.radians(180.0)
        self.lock_center = rospy.get_param("~lock_center", True)

        self.target_cw_positive = True  # 目标角正值表示顺时针
        target_init_deg = rospy.get_param("~target_yaw_deg", 0.0)
        self.target_yaw = wrap_pi(self._deg_to_target_rad(target_init_deg))

        # 状态
        self.have_pose = False
        self.pose_msg = PoseStamped()
        self.current_yaw = 0.0
        self.target_received = not self.require_target_input
        self.int_err = 0.0
        self.prev_err = 0.0
        self.d_filt = 0.0
        self.last_time = None
        self.last_yaw_rate_cmd = 0.0
        self.prev_yaw_for_rate = 0.0
        self.yaw_rate_meas_filt = 0.0
        self.center_locked = False
        self.center_x = 0.0
        self.center_y = 0.0
        self.center_z = 0.0
        self.tip_trail = deque(maxlen=self.trail_max_points)
        self.last_param_check_time = rospy.Time(0)
        self.param_check_period = max(0.05, rospy.get_param("~param_check_period", 0.2))

        # ROS IO
        self.cmd_pub = rospy.Publisher(self.cmd_topic, VelCmd, queue_size=1)
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)
        self.cur_yaw_pub = rospy.Publisher(self.cur_yaw_topic, Float64, queue_size=1)
        self.err_yaw_pub = rospy.Publisher(self.err_yaw_topic, Float64, queue_size=1)
        self.cmd_yaw_pub = rospy.Publisher(self.cmd_yaw_topic, Float64, queue_size=1)

        rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber(self.target_topic, Float64, self.target_cb, queue_size=1)

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo(
            "yaw_tuner started: mode=%s target=%.1f deg",
            "basic_dev" if self.use_basic_dev_yaw_logic else "pid",
            math.degrees(self.target_yaw),
        )
        if self.use_basic_dev_yaw_logic:
            rospy.loginfo(
                "basic_dev yaw: kp=%.2f kd_rate=%.2f pure_p=%d yaw_soft_limit_deg=%.1f stop_tol=%.2f disable_limit=%d disable_tol=%d max_yaw_accel=%.1f",
                self.fp_kp_yaw,
                self.fp_kd_yaw_rate,
                int(self.pure_p_mode),
                self.yaw_soft_limit_deg,
                self.yaw_stop_tolerance_deg,
                int(self.disable_yaw_limit),
                int(self.disable_yaw_stop_tolerance),
                self.max_yaw_accel_deg_s2,
            )
        else:
            rospy.loginfo(
                "pid yaw: kp=%.2f ki=%.2f kd=%.2f max=%.1f",
                self.kp,
                self.ki,
                self.kd,
                self.max_yaw_rate_deg,
            )
        if self.require_target_input:
            rospy.loginfo("yaw_tuner waiting target input on %s before rotating.", self.target_topic)
        rospy.loginfo("cmd_cw_positive=%d invert_pose_yaw=%d target_cw_positive=%d yaw_invert_direction=%d viz_invert_yaw=%d viz_yaw_offset_deg=%.1f",
                      int(self.cmd_cw_positive), int(self.invert_pose_yaw),
                      int(self.target_cw_positive), int(self.yaw_invert_direction), int(self.viz_invert_yaw),
                      math.degrees(self.viz_yaw_offset_rad))

    def _deg_to_target_rad(self, deg):
        """将外部目标角（默认正=顺时针）转换为内部CCW正的弧度"""
        rad = math.radians(deg)
        return -rad if self.target_cw_positive else rad

    def _maybe_update_runtime_params(self, now):
        """运行时热更新关键参数，支持 rosparam set 实时调参。"""
        if (now - self.last_param_check_time).to_sec() < self.param_check_period:
            return
        self.last_param_check_time = now

        watched_params = [
            ("fp_kp_yaw", float),
            ("fp_kd_yaw_rate", float),
            ("yaw_soft_limit_deg", float),
            ("max_yaw_accel_deg_s2", float),
            ("yaw_stop_tolerance_deg", float),
            ("yaw_min_cmd_deg", float),
            ("yaw_rate_lpf_alpha", float),
            ("pure_p_mode", bool),
            ("disable_yaw_limit", bool),
            ("disable_yaw_stop_tolerance", bool),
        ]

        changed = []
        for name, caster in watched_params:
            old_val = getattr(self, name)
            new_val = rospy.get_param("~" + name, old_val)
            try:
                new_val = caster(new_val)
            except (TypeError, ValueError):
                continue

            if isinstance(old_val, float):
                if abs(new_val - old_val) <= 1e-9:
                    continue
            elif new_val == old_val:
                continue

            setattr(self, name, new_val)
            changed.append((name, old_val, new_val))

        if changed:
            self.yaw_rate_lpf_alpha = clamp(self.yaw_rate_lpf_alpha, 0.0, 1.0)
            msg = ", ".join(["{}:{}->{}".format(k, o, n) for k, o, n in changed])
            rospy.loginfo("参数更新：%s", msg)

    def pose_cb(self, msg):
        self.pose_msg = msg
        yaw = quat_to_yaw(msg.pose.orientation)
        if self.invert_pose_yaw:
            yaw = -yaw
        self.current_yaw = wrap_pi(yaw)
        self.have_pose = True

    def target_cb(self, msg):
        self.target_yaw = wrap_pi(self._deg_to_target_rad(msg.data))
        self.target_received = True
        rospy.loginfo("New target yaw: %.2f deg", msg.data)

    def publish_cmd(self, yaw_rate_deg):
        cmd = VelCmd()
        cmd.header.stamp = rospy.Time.now()
        cmd.vx = 0.0
        cmd.vy = 0.0
        cmd.vz = 0.0
        cmd.yawRate = yaw_rate_deg
        cmd.va = 8
        cmd.stop = 0
        self.cmd_pub.publish(cmd)

    def publish_markers(self, yaw_err_deg, yaw_rate_deg):
        if not self.have_pose:
            return

        frame_id = self.pose_msg.header.frame_id if self.pose_msg.header.frame_id else self.viz_frame_id
        if self.lock_center:
            if not self.center_locked:
                self.center_x = self.pose_msg.pose.position.x
                self.center_y = self.pose_msg.pose.position.y
                self.center_z = self.pose_msg.pose.position.z
                self.center_locked = True
            cx = self.center_x
            cy = self.center_y
            cz = self.center_z + self.arrow_z_up
        else:
            cx = self.pose_msg.pose.position.x
            cy = self.pose_msg.pose.position.y
            cz = self.pose_msg.pose.position.z + self.arrow_z_up

        # 竞赛地图通常使用NED高度（z为负）时，默认把可视化抬到固定平面，避免“有数据但看不见”。
        if self.draw_on_plane:
            cz = self.draw_plane_z

        cur_yaw_vis = self.current_yaw
        tar_yaw_vis = self.target_yaw
        if self.viz_invert_yaw:
            cur_yaw_vis = -cur_yaw_vis
            tar_yaw_vis = -tar_yaw_vis
        cur_yaw_vis = wrap_pi(cur_yaw_vis + self.viz_yaw_offset_rad)
        tar_yaw_vis = wrap_pi(tar_yaw_vis + self.viz_yaw_offset_rad)

        cur_end = Point(
            x=cx + self.arrow_len * math.cos(cur_yaw_vis),
            y=cy + self.arrow_len * math.sin(cur_yaw_vis),
            z=cz,
        )
        tar_end = Point(
            x=cx + self.arrow_len * math.cos(tar_yaw_vis),
            y=cy + self.arrow_len * math.sin(tar_yaw_vis),
            z=cz,
        )

        m_cur = Marker()
        m_cur.header.frame_id = frame_id
        m_cur.header.stamp = rospy.Time.now()
        m_cur.ns = "yaw_tuner"
        m_cur.id = 0
        m_cur.type = Marker.ARROW
        m_cur.action = Marker.ADD
        m_cur.scale.x = 0.08
        m_cur.scale.y = 0.14
        m_cur.scale.z = 0.18
        m_cur.color.a = 1.0
        m_cur.color.g = 1.0
        m_cur.pose.orientation.w = 1.0
        m_cur.points = [Point(x=cx, y=cy, z=cz), cur_end]

        m_tar = Marker()
        m_tar.header.frame_id = frame_id
        m_tar.header.stamp = rospy.Time.now()
        m_tar.ns = "yaw_tuner"
        m_tar.id = 1
        # 目标角使用红色线段，和当前角绿色箭头做区分
        m_tar.type = Marker.LINE_STRIP
        m_tar.action = Marker.ADD
        m_tar.scale.x = 0.06
        m_tar.color.a = 1.0
        m_tar.color.r = 1.0
        m_tar.pose.orientation.w = 1.0
        m_tar.points = [Point(x=cx, y=cy, z=cz), tar_end]

        m_txt = Marker()
        m_txt.header.frame_id = frame_id
        m_txt.header.stamp = rospy.Time.now()
        m_txt.ns = "yaw_tuner"
        m_txt.id = 2
        m_txt.type = Marker.TEXT_VIEW_FACING
        m_txt.action = Marker.ADD
        m_txt.pose.position.x = cx
        m_txt.pose.position.y = cy
        m_txt.pose.position.z = cz + 0.35
        m_txt.pose.orientation.w = 1.0
        m_txt.scale.z = 0.25
        m_txt.color.a = 1.0
        m_txt.color.r = 1.0
        m_txt.color.g = 1.0
        m_txt.color.b = 1.0
        err_vis_deg = math.degrees(wrap_pi(tar_yaw_vis - cur_yaw_vis))
        m_txt.text = "cur={:.1f} tgt={:.1f} err={:.1f} cmd={:.1f}".format(
            math.degrees(cur_yaw_vis), math.degrees(tar_yaw_vis), err_vis_deg, yaw_rate_deg
        )

        # 记录当前角箭头端点轨迹，用于观察摇摆/振荡
        if self.show_trail:
            if not self.tip_trail:
                self.tip_trail.append(Point(x=cur_end.x, y=cur_end.y, z=cur_end.z))
            else:
                last = self.tip_trail[-1]
                dist = math.sqrt(
                    (cur_end.x - last.x) ** 2 + (cur_end.y - last.y) ** 2 + (cur_end.z - last.z) ** 2
                )
                if dist >= self.trail_min_step:
                    self.tip_trail.append(Point(x=cur_end.x, y=cur_end.y, z=cur_end.z))

        m_trail = Marker()
        m_trail.header.frame_id = frame_id
        m_trail.header.stamp = rospy.Time.now()
        m_trail.ns = "yaw_tuner"
        m_trail.id = 3
        m_trail.type = Marker.LINE_STRIP
        m_trail.action = Marker.ADD
        m_trail.scale.x = 0.03
        m_trail.color.a = 0.95
        m_trail.color.r = 0.1
        m_trail.color.g = 0.9
        m_trail.color.b = 1.0
        m_trail.pose.orientation.w = 1.0
        m_trail.points = list(self.tip_trail)

        arr = MarkerArray()
        arr.markers = [m_cur, m_tar, m_txt, m_trail]
        self.marker_pub.publish(arr)

    def step(self):
        now = rospy.Time.now()
        self._maybe_update_runtime_params(now)

        if not self.have_pose:
            self.publish_cmd(0.0)
            rospy.loginfo_throttle(1.0, "yaw_tuner waiting pose...")
            return

        if self.last_time is None:
            self.last_time = now
            self.prev_err = wrap_pi(self.target_yaw - self.current_yaw)
            self.prev_yaw_for_rate = self.current_yaw
            self.yaw_rate_meas_filt = 0.0
            self.publish_cmd(0.0)
            return

        # 按需求：未收到手动目标角之前，始终保持原地不转
        if not self.target_received:
            self.target_yaw = self.current_yaw
            self.publish_cmd(0.0)
            self.publish_markers(0.0, 0.0)
            self.cur_yaw_pub.publish(Float64(data=math.degrees(self.current_yaw)))
            self.err_yaw_pub.publish(Float64(data=0.0))
            self.cmd_yaw_pub.publish(Float64(data=0.0))
            rospy.loginfo_throttle(1.0, "yaw_tuner waiting manual target...")
            return

        dt = clamp((now - self.last_time).to_sec(), 1e-3, 0.1)
        # 误差定义（内部统一用 CCW 正）：err = target - current
        # err>0 表示需要“逆时针(CCW)”转；err<0 表示需要“顺时针(CW)”转
        # 如果外部目标是 CW 正（target_cw_positive=true），在 target_cb 已转换为内部CCW定义。
        err = wrap_pi(self.target_yaw - self.current_yaw)
        err_deg = math.degrees(err)
        rospy.loginfo_throttle(
            0.2,
            "【Yaw PID】target=%.1f cur=%.1f err=%.1f deg",
            math.degrees(self.target_yaw),
            math.degrees(self.current_yaw),
            err_deg,
        )

        # 角速度反馈：由姿态差分估计当前 yaw 角速度（deg/s）
        yaw_delta = wrap_pi(self.current_yaw - self.prev_yaw_for_rate)
        yaw_rate_meas_raw = math.degrees(yaw_delta) / dt
        self.yaw_rate_meas_filt = (
            (1.0 - self.yaw_rate_lpf_alpha) * self.yaw_rate_meas_filt
            + self.yaw_rate_lpf_alpha * yaw_rate_meas_raw
        )
        self.prev_yaw_for_rate = self.current_yaw

        p_term = 0.0
        d_term = 0.0
        sum_term = 0.0

        if self.use_basic_dev_yaw_logic:
            if self.pure_p_mode:
                # 纯P：用于快速确认正负方向与可视化一致性
                p_term = self.fp_kp_yaw * err_deg
                d_term = 0.0
                sum_term = p_term
                yaw_ccw_deg = sum_term
            else:
                # PD（速度反馈D）：yawRate = Kp*e - Kd*omega_meas
                # P项符号影响方向：当 err>0（应逆时针）时，若 kp>0 则 P项为正，驱动 CCW。
                # D项是阻尼：当前角速度与目标旋转方向相同则抵消输出，抑制过冲。
                if (not self.disable_yaw_stop_tolerance) and (abs(err_deg) <= self.yaw_stop_tolerance_deg):
                    p_term = self.fp_kp_yaw * err_deg
                    d_term = -self.fp_kd_yaw_rate * self.yaw_rate_meas_filt
                    sum_term = 0.0
                    yaw_ccw_deg = 0.0
                else:
                    p_term = self.fp_kp_yaw * err_deg
                    d_term = -self.fp_kd_yaw_rate * self.yaw_rate_meas_filt
                    sum_term = p_term + d_term
                    yaw_ccw_deg = sum_term
                    if not self.disable_yaw_limit:
                        yaw_ccw_deg = clamp(
                            yaw_ccw_deg,
                            -self.yaw_soft_limit_deg,
                            self.yaw_soft_limit_deg,
                        )
            # cmd_cw_positive 的作用：
            # - True: 发布给飞控的 yawRate 采用“顺时针为正”，所以内部CCW量要取反
            # - False: 发布给飞控的 yawRate 采用“逆时针为正”，与内部定义同向
            yaw_rate_cmd = -yaw_ccw_deg if self.cmd_cw_positive else yaw_ccw_deg
        else:
            deadband = math.radians(self.deadband_deg)
            if abs(err) < deadband:
                err = 0.0
                self.int_err *= 0.95

            self.int_err = clamp(self.int_err + err * dt, -self.i_limit, self.i_limit)
            d_raw = (err - self.prev_err) / dt
            self.d_filt = (1.0 - self.d_lpf_alpha) * self.d_filt + self.d_lpf_alpha * d_raw

            u_ccw = self.kp * err + self.ki * self.int_err + self.kd * self.d_filt
            u_deg = clamp(math.degrees(u_ccw), -self.max_yaw_rate_deg, self.max_yaw_rate_deg)
            yaw_rate_cmd = -u_deg if self.cmd_cw_positive else u_deg
            p_term = math.degrees(self.kp * err)
            d_term = math.degrees(self.kd * self.d_filt)
            sum_term = u_deg

        # 可选的方向反转开关：当平台方向定义与预期完全反向时使用
        if self.yaw_invert_direction:
            yaw_rate_cmd = -yaw_rate_cmd

        # 方向一致性检查（按“发布指令的符号定义”进行检查）
        err_for_cmd_sign_deg = -err_deg if self.cmd_cw_positive else err_deg
        if self.yaw_invert_direction:
            err_for_cmd_sign_deg = -err_for_cmd_sign_deg

        # 角速度斜坡限制：限制相邻控制周期的 yawRate 变化量，避免过冲振荡
        if (not self.pure_p_mode) and (self.max_yaw_accel_deg_s2 > 0.0):
            max_delta = self.max_yaw_accel_deg_s2 * dt
            yaw_rate_cmd = clamp(
                yaw_rate_cmd,
                self.last_yaw_rate_cmd - max_delta,
                self.last_yaw_rate_cmd + max_delta,
            )

        # 接近目标时，若还未进入停止容差但输出太小，给一个最小角速度避免“爬行”
        if (not self.disable_yaw_stop_tolerance) and (abs(err_deg) > self.yaw_stop_tolerance_deg):
            if abs(yaw_rate_cmd) < self.yaw_min_cmd_deg:
                yaw_rate_cmd = self.yaw_min_cmd_deg if err_for_cmd_sign_deg >= 0.0 else -self.yaw_min_cmd_deg

        rospy.loginfo_throttle(
            0.2,
            "【PID项】P=%.2f D=%.2f sum=%.2f cmd=%.2f",
            p_term,
            d_term,
            sum_term,
            yaw_rate_cmd,
        )

        if err_for_cmd_sign_deg > 0.0 and yaw_rate_cmd < 0.0:
            rospy.logwarn_throttle(0.5, "方向警告：误差为正但输出为负，err_cmd=%.2f cmd=%.2f", err_for_cmd_sign_deg, yaw_rate_cmd)
        if err_for_cmd_sign_deg < 0.0 and yaw_rate_cmd > 0.0:
            rospy.logwarn_throttle(0.5, "方向警告：误差为负但输出为正，err_cmd=%.2f cmd=%.2f", err_for_cmd_sign_deg, yaw_rate_cmd)
        if err_for_cmd_sign_deg * yaw_rate_cmd < 0.0:
            rospy.logwarn_throttle(0.5, "⚠️ 方向警告：误差和输出符号不一致！检查 kp 符号 / cmd_cw_positive / yaw_invert_direction")

        self.publish_cmd(yaw_rate_cmd)
        self.publish_markers(err_deg, yaw_rate_cmd)
        self.cur_yaw_pub.publish(Float64(data=math.degrees(self.current_yaw)))
        self.err_yaw_pub.publish(Float64(data=err_deg))
        self.cmd_yaw_pub.publish(Float64(data=yaw_rate_cmd))

        self.prev_err = err
        self.last_yaw_rate_cmd = yaw_rate_cmd
        self.last_time = now

        rospy.loginfo_throttle(
            0.2,
            "yaw cur=%.1f tgt=%.1f err=%.1f w=%.1f cmd=%.1f",
            math.degrees(self.current_yaw),
            math.degrees(self.target_yaw),
            err_deg,
            self.yaw_rate_meas_filt,
            yaw_rate_cmd,
        )

    def run(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()

    def on_shutdown(self):
        self.publish_cmd(0.0)
        rospy.sleep(0.05)
        self.publish_cmd(0.0)


if __name__ == "__main__":
    try:
        YawTuner().run()
    except rospy.ROSInterruptException:
        pass
