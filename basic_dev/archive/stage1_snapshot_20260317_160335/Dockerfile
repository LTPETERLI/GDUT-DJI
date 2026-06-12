FROM osrf/ros:noetic-desktop-full-focal

# 拷贝源码与轨迹文件
ADD src /basic_dev/src/
ADD gnss_path.csv /basic_dev/gnss_path.csv
ADD gnss_path_fast.csv /basic_dev/gnss_path_fast.csv
ADD gnss_path_fast_aligned.csv /basic_dev/gnss_path_fast_aligned.csv
ADD 赛道3反向_smoothed_yflip.csv /basic_dev/赛道3反向_smoothed_yflip.csv
ADD setup.bash /

RUN chmod +x /setup.bash

USER root

RUN apt update && apt install -y python3-catkin-tools ros-noetic-geographic-msgs \
 ros-noetic-tf2-sensor-msgs ros-noetic-tf2-geometry-msgs ros-noetic-image-transport \
 net-tools

ENV ROS_DISTRO=noetic

WORKDIR /basic_dev/
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && catkin_make

ENTRYPOINT [ "/setup.bash" ]
