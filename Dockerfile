# ============================================================================
# ROS2 Autonomous Explorer — Docker image
# Pins ROS2 Humble + Gazebo Fortress + all dependencies in one reproducible image.
#
# Build:   docker build -t ros2-explorer .
# Run:     docker run --rm -it ros2-explorer
# Train:   docker run --rm -it --gpus all ros2-explorer \
#            bash -c "ros2 launch rl_local_planner train.launch.py headless:=true &
#                     sleep 25 && python3 src/rl_local_planner/scripts/train_ppo.py"
# ============================================================================

FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-xacro \
    git \
    wget \
    # Gazebo Fortress (Ignition)
    libignition-fortress-dev \
    ros-humble-ros-gz \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-image \
    # Nav2
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    # SLAM & localization
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    ros-humble-imu-filter-madgwick \
    # Robot description
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-xacro \
    # Misc ROS2
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── Copy project source ──────────────────────────────────────────────────────
WORKDIR /ros2_ws
COPY . /ros2_ws/

# ── Build the ROS2 workspace ─────────────────────────────────────────────────
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY <<'ENTRYPOINT_SCRIPT' /ros2_ws/entrypoint.sh
#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
exec "$@"
ENTRYPOINT_SCRIPT
RUN chmod +x /ros2_ws/entrypoint.sh

ENTRYPOINT ["/ros2_ws/entrypoint.sh"]
CMD ["bash"]
