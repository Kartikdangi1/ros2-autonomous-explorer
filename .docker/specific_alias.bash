#!/usr/bin/env bash
alias cb='colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'
alias cbc='colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select'
alias ct='colcon test'
alias cs='source /ros2_ws/install/setup.bash'

alias explore='ros2 launch autonomous_explorer nav2_exploration.launch.py use_rviz:=true'
alias explore_rl='ros2 launch autonomous_explorer nav2_exploration.launch.py controller:=rl use_rviz:=true'
alias train='python3 src/rl_local_planner/scripts/train_ppo.py'
alias tb='tensorboard --logdir ./tb_logs/'
