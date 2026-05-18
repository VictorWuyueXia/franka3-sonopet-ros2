cmake_minimum_required(VERSION 3.8)
project(fr3_sonopet_interfaces)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
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

ament_export_dependencies(rosidl_default_runtime)
ament_package()
