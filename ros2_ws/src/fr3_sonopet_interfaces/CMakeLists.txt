cmake_minimum_required(VERSION 3.8)
project(fr3_sonopet_interfaces)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rviz_common REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RasterPatch.msg"
  "msg/RasterPlan.msg"
  "msg/RunState.msg"
  "msg/AudioChunk.msg"
  "action/BuildRasterPlan.action"
  "action/PreviewMotion.action"
  "action/ExecuteMotion.action"
  "action/RecordExperiment.action"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_library(motion_control_panel SHARED src/motion_control_panel.cpp)
target_link_libraries(motion_control_panel Qt5::Widgets "${cpp_typesupport_target}")
ament_target_dependencies(motion_control_panel
  pluginlib
  rclcpp
  rclcpp_action
  rviz_common
)
pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

install(TARGETS motion_control_panel
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
