cmake_minimum_required(VERSION 3.0.2)
project(basic_dev)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
set(pcl_conversions_DIR /usr/local/share/pcl_conversions/cmake)
set(pcl_ros_DIR /usr/local/share/pcl_ros/cmake)
find_package(catkin REQUIRED COMPONENTS
airsim_ros
message_generation
nav_msgs
visualization_msgs
# nodelet
roscpp
rospy
sensor_msgs
std_msgs
geographic_msgs
geometry_msgs
std_srvs
tf2
tf2_ros
tf2_sensor_msgs
tf2_geometry_msgs
image_transport
cv_bridge
pcl_ros
pcl_conversions
)
find_package(OpenCV REQUIRED)
find_package(X11 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS airsim_ros message_runtime roscpp std_msgs # nodelet 
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${X11_INCLUDE_DIR})


add_executable(basic_dev src/basic_dev.cpp )
target_link_libraries(basic_dev ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(kb_ctrl src/kb_ctrl.cpp)
target_link_libraries(kb_ctrl ${catkin_LIBRARIES} ${X11_LIBRARIES})

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  # PATTERN ".git" EXCLUDE
)

catkin_install_python(
  PROGRAMS
    scripts/yaw_tuner.py
    scripts/log_flight.py
    scripts/visualize_stage_paths.py
    scripts/parse_stage2_log.py
    scripts/record_waypoints.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
