#!/bin/bash
set -e

source /opt/ros/noetic/setup.bash
cd /basic_dev
source devel/setup.bash

exec rosrun basic_dev basic_dev
