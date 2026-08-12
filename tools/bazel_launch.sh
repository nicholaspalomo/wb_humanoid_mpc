#!/bin/bash
# Wrapper script to run ROS2 launch files with Bazel-built binaries

if [ "$#" -lt 1 ]; then
    echo "Usage: ./bazel_launch.sh <package_name> <launch_file.launch.py> [args...]"
    exit 1
fi

PACKAGE_NAME=$1
LAUNCH_FILE=$2
shift 2

# Source ROS2 underlay
source /opt/ros/$ROS_DISTRO/setup.bash

# Ensure we're in the workspace root
WORKSPACE_ROOT=$(git rev-parse --show-toplevel)
cd $WORKSPACE_ROOT

# Build the required packages if not already built (optional, can be commented out)
# bazel build //...

# Add Bazel output directories to AMENT_PREFIX_PATH so ROS2 can find the binaries
# and shared libraries.
export AMENT_PREFIX_PATH=$WORKSPACE_ROOT/bazel-bin:$AMENT_PREFIX_PATH
export LD_LIBRARY_PATH=$WORKSPACE_ROOT/bazel-bin:$LD_LIBRARY_PATH

echo "Running ros2 launch $PACKAGE_NAME $LAUNCH_FILE $@"
ros2 launch $PACKAGE_NAME $LAUNCH_FILE $@
