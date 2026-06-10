cmake_minimum_required(VERSION 3.18)
project(ffs_ros LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type: Debug, Release, RelWithDebInfo" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core)
find_package(CUDAToolkit REQUIRED)

# Reuse the canonical C++ TensorRT runtime build from cpp/. The package must
# live inside the Fast-FoundationStereo repo, or FFS_CPP_DIR must be set.
set(FFS_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../cpp" CACHE PATH
    "Path to the Fast-FoundationStereo cpp/ sources")
if(NOT EXISTS "${FFS_CPP_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Fast-FoundationStereo cpp/ sources not found at ${FFS_CPP_DIR}. "
                      "Set -DFFS_CPP_DIR=/path/to/Fast-FoundationStereo/cpp")
endif()
add_subdirectory(${FFS_CPP_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ffs_cpp)

add_executable(ffs_stereo_node src/ffs_stereo_node.cpp)
target_link_libraries(ffs_stereo_node
  ffs_depth_inference
  ${OpenCV_LIBS}
  CUDA::cudart
)
ament_target_dependencies(ffs_stereo_node
  rclcpp
  sensor_msgs
  cv_bridge
  message_filters
)

install(TARGETS ffs_stereo_node DESTINATION lib/${PROJECT_NAME})

# Also expose the cpp/ tools through `ros2 run ffs_ros ...` so engines can be
# built inside the runtime container (engines are GPU- and TRT-version-specific).
install(TARGETS ffs_build_single_engine DESTINATION lib/${PROJECT_NAME})
install(TARGETS ffs_gwc_plugin LIBRARY DESTINATION lib)
if(TARGET ffs_depth_main)
  install(TARGETS ffs_depth_main ffs_profile_speed DESTINATION lib/${PROJECT_NAME})
endif()

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
