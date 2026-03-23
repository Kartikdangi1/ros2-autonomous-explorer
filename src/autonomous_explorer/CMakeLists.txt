cmake_minimum_required(VERSION 3.10)
project(autonomous_explorer)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/obstacle_cluster_node.py
  scripts/nbv_goal_provider_node.py
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
  urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
  OPTIONAL
)

ament_package()
