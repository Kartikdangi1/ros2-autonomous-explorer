# ============================================================================
# ros2-explorer — Multi-stage Dockerfile
#
# Stages:
#   base_ws  — ROS2 Humble + Gazebo Fortress + Nav2 + SLAM + localization
#   user_ws  — Python ML deps (torch, SB3, gymnasium) + workspace build
#   dev      — Non-root user matched to host UID/GID, VS Code server
#
# Build: ./container.bash -b
# ============================================================================

# ── Stage 1: base_ws ─────────────────────────────────────────────────────────
FROM ros:humble-ros-base-jammy AS base_ws

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-xacro \
    git wget curl \
    libignition-fortress-dev \
    ros-humble-ros-gz \
    ros-humble-ros-gz-bridge \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-image \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    ros-humble-imu-filter-madgwick \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-xacro \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-rviz2 \
    && rosdep init || true && rosdep update

# ── Stage 2: user_ws ─────────────────────────────────────────────────────────
FROM base_ws AS user_ws

ARG TORCH=false

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r /tmp/requirements.txt

# Conditionally install GPU torch (large layer — skip unless TORCH=true)
RUN if [ "$TORCH" = "true" ]; then \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118; \
    fi

WORKDIR /ros2_ws
COPY src/ /ros2_ws/src/
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/ros2_ws/build \
    source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build \
        --symlink-install \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --parallel-workers 4

# ── Stage 3: dev ─────────────────────────────────────────────────────────────
FROM user_ws AS dev

ARG USER=devuser
ARG UID=1000
ARG GID=1000

RUN groupadd --gid ${GID} ${USER} \
    && useradd --uid ${UID} --gid ${GID} --create-home ${USER} \
    && echo "${USER} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USER} \
    && chmod 440 /etc/sudoers.d/${USER}

RUN mkdir /commandhistory && touch /commandhistory/.bash_history \
    && chown -R ${USER}:${USER} /commandhistory

RUN echo "source /opt/ros/humble/setup.bash" >> /home/${USER}/.bashrc && \
    echo "source /ros2_ws/install/setup.bash 2>/dev/null || true" >> /home/${USER}/.bashrc && \
    echo "source /ros2_ws/.docker/specific_alias.bash 2>/dev/null || true" >> /home/${USER}/.bashrc && \
    echo "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
        >> /home/${USER}/.bashrc

USER ${USER}
RUN mkdir -p /home/${USER}/.vscode-server/data/Machine \
    && curl -fsSL https://raw.githubusercontent.com/b01/dl-vscode-server/refs/tags/0.2.3/download-vs-code.sh \
        -o /tmp/dl.sh && chmod +x /tmp/dl.sh && bash /tmp/dl.sh "linux" "x64" && rm /tmp/dl.sh

USER root
RUN chown -R ${USER}:${USER} /ros2_ws

COPY <<'EOF' /entrypoint.bash
#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash 2>/dev/null || true
exec "$@"
EOF
RUN chmod +x /entrypoint.bash

USER ${USER}
WORKDIR /ros2_ws
ENTRYPOINT ["/entrypoint.bash"]
CMD ["bash"]
