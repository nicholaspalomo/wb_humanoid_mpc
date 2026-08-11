cmake_minimum_required(VERSION 3.5)
project(ocs2_ros2_msgs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unique_identifier_msgs REQUIRED)

set(msg_files
  "msg/Benchmarks.idl"
  "msg/BezierSegment.idl"
  "msg/BezierTrajectory.idl"
  "msg/Constraint.idl"
  "msg/ControllerData.idl"
  "msg/Gains.idl"
  "msg/IndividualBenchmarks.idl"
  "msg/IndividualGains.idl"
  "msg/LagrangianMetrics.idl"
  "msg/ModeSchedule.idl"
  "msg/MpcFlattenedController.idl"
  "msg/MpcInput.idl"
  "msg/MpcObservation.idl"
  "msg/MpcPerformanceIndices.idl"
  "msg/MpcState.idl"
  "msg/MpcTargetTrajectories.idl"
  "msg/Multiplier.idl"
  "msg/ThreeDofPath.idl"
  "msg/ThreeDofPointTS.idl"
)

set(srv_files
  "srv/Reset.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES rcl_interfaces builtin_interfaces geometry_msgs std_msgs unique_identifier_msgs
  ADD_LINTER_TESTS
)

ament_package()
