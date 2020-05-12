# vcpkg cmake version
cmake_minimum_required(VERSION 3.14)
project(pnp)

option(BUILD_TEST "Build test" ON)

find_package(Eigen3 CONFIG REQUIRED)
find_package(OpenCV CONFIG)
if(OpenCV_NOT_FOUND)
set(BUILD_TEST OFF)
endif(OpenCV_NOT_FOUND)

set(CMAKE_CXX_STANDARD_REQUIRED 17)
set(CMAKE_BUILD_TYPE Release)

include_directories("/home/qy101/Desktop/cppproject/pnp")

if(BUILD_TEST)
add_executable(pnp test/main.cpp src/pnp_gn.cpp src/pnp_dlt.cpp src/epnp.cpp)
target_link_libraries(pnp PRIVATE quirc opencv_ml opencv_dnn opencv_dpm Eigen3::Eigen)
endif(BUILD_TEST)
add_library(pnp_gn src/pnp_gn.cpp)
target_link_libraries(pnp_gn PRIVATE Eigen3::Eigen)
add_library(pnp_dlt src/pnp_dlt.cpp)
target_link_libraries(pnp_dlt PRIVATE Eigen3::Eigen)
add_library(epnp src/epnp.cpp)
target_link_libraries(epnp PRIVATE Eigen3::Eigen)
