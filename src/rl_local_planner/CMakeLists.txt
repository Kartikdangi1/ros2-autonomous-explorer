cmake_minimum_required(VERSION 3.10)
project(rl_local_planner)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/rl_controller_node.py
  scripts/train_ppo.py
  scripts/export_onnx.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
  OPTIONAL
)

install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
  OPTIONAL
)

install(DIRECTORY
  models/
  DESTINATION share/${PROJECT_NAME}/models
  OPTIONAL
)

ament_package()
