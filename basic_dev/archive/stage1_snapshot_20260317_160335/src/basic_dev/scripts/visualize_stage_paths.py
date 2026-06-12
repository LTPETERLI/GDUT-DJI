#!/usr/bin/env python3
import csv
from pathlib import Path

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg
from visualization_msgs.msg import Marker, MarkerArray


def load_xyz(csv_path):
    pts = []
    with open(csv_path, "r", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if len(row) >= 4:
                # 兼容 time,x,y,z 和 time,x,y,z,qx,qy,qz,qw
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
            elif len(row) >= 3:
                x = float(row[0])
                y = float(row[1])
                z = float(row[2])
            else:
                continue
            pts.append((x, y, z))
    return pts


def make_path(points, frame_id):
    msg = PathMsg()
    now = rospy.Time.now()
    msg.header.stamp = now
    msg.header.frame_id = frame_id
    for x, y, z in points:
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = frame_id
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg


def make_endpoint_markers(points, ns, frame_id, rgb):
    arr = MarkerArray()
    if not points:
        return arr
    labels = [("start", points[0]), ("end", points[-1])]
    for idx, (label, (x, y, z)) in enumerate(labels):
        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = frame_id
        m.ns = ns
        m.id = idx
        m.action = Marker.ADD
        m.type = Marker.SPHERE
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0
        m.scale.x = 0.8
        m.scale.y = 0.8
        m.scale.z = 0.8
        m.color.a = 1.0
        m.color.r = rgb[0]
        m.color.g = rgb[1]
        m.color.b = rgb[2]
        arr.markers.append(m)

        t = Marker()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = frame_id
        t.ns = ns + "_text"
        t.id = idx
        t.action = Marker.ADD
        t.type = Marker.TEXT_VIEW_FACING
        t.pose.position.x = x
        t.pose.position.y = y
        t.pose.position.z = z + 1.2
        t.pose.orientation.w = 1.0
        t.scale.z = 0.9
        t.color.a = 1.0
        t.color.r = rgb[0]
        t.color.g = rgb[1]
        t.color.b = rgb[2]
        t.text = f"{ns}_{label}"
        arr.markers.append(t)
    return arr


def resolve_default_stage2():
    base = Path("/home/peter/RMUA2026-LT/IntelligentUAVChampionshipBase-RMUA2026/basic_dev")
    candidates = [
        base / "赛道3反向_smoothed_yflip.csv",
        base / "赛道3反向_smoothed.csv",
        base / "赛道3反向.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


def main():
    rospy.init_node("visualize_stage_paths")
    frame_id = rospy.get_param("~frame_id", "map")
    stage1_csv = rospy.get_param(
        "~stage1_csv",
        "/home/peter/RMUA2026-LT/IntelligentUAVChampionshipBase-RMUA2026/basic_dev/gnss_path_fast_aligned.csv",
    )
    stage2_csv = rospy.get_param("~stage2_csv", resolve_default_stage2())

    stage1_pts = load_xyz(stage1_csv)
    stage2_pts = load_xyz(stage2_csv)

    pub1 = rospy.Publisher("/viz/stage1_path", PathMsg, queue_size=1, latch=True)
    pub2 = rospy.Publisher("/viz/stage2_path", PathMsg, queue_size=1, latch=True)
    marker_pub = rospy.Publisher("/viz/stage_path_markers", MarkerArray, queue_size=1, latch=True)

    pub1.publish(make_path(stage1_pts, frame_id))
    pub2.publish(make_path(stage2_pts, frame_id))

    markers = MarkerArray()
    markers.markers.extend(make_endpoint_markers(stage1_pts, "stage1", frame_id, (1.0, 0.85, 0.1)).markers)
    markers.markers.extend(make_endpoint_markers(stage2_pts, "stage2", frame_id, (0.1, 0.8, 1.0)).markers)
    marker_pub.publish(markers)

    rospy.loginfo("Published stage paths: stage1=%s (%d pts), stage2=%s (%d pts)",
                  stage1_csv, len(stage1_pts), stage2_csv, len(stage2_pts))
    rospy.spin()


if __name__ == "__main__":
    main()
