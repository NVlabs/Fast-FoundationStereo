#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
if [ -f /opt/ffs/ros2/install/setup.bash ]; then
  source /opt/ffs/ros2/install/setup.bash
fi
exec "$@"
