#!/usr/bin/env python3
# 发布简单无人机模型marker,挂在drone_follow,RViz订阅/viz/drone_model显示
import rospy
from visualization_msgs.msg import Marker, MarkerArray

def make(ns, mid, mtype, sx,sy,sz, px,py,pz, r,g,b, frame):
    m=Marker()
    m.header.frame_id=frame; m.ns=ns; m.id=mid; m.type=mtype; m.action=Marker.ADD
    m.scale.x=sx; m.scale.y=sy; m.scale.z=sz
    m.pose.position.x=px; m.pose.position.y=py; m.pose.position.z=pz
    m.pose.orientation.w=1.0
    m.color.r=r; m.color.g=g; m.color.b=b; m.color.a=0.95
    return m

rospy.init_node('drone_marker')
frame=rospy.get_param('~frame','drone_follow')
pub=rospy.Publisher('/viz/drone_model', MarkerArray, queue_size=1)
rate=rospy.Rate(20)
arm=0.18  # 轴距
while not rospy.is_shutdown():
    arr=MarkerArray()
    # 机身(中心立方)
    arr.markers.append(make('body',0,Marker.CUBE, 0.18,0.18,0.08, 0,0,0, 0.1,0.1,0.1, frame))
    # 4个旋翼(圆柱),x前为机头
    rotors=[(arm,arm,1.0,0.3,0.3),(arm,-arm,0.3,0.3,1.0),(-arm,arm,1.0,1.0,0.3),(-arm,-arm,0.3,1.0,0.3)]
    for i,(x,y,r,g,b) in enumerate(rotors):
        arr.markers.append(make('rotor',i+1,Marker.CYLINDER, 0.20,0.20,0.03, x,y,0.04, r,g,b, frame))
    # 机头方向箭头(红,指+x)
    arr.markers.append(make('head',10,Marker.ARROW, 0.35,0.05,0.05, 0,0,0, 1.0,0,0, frame))
    pub.publish(arr)
    rate.sleep()
