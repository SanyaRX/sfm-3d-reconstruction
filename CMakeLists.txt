cmake_minimum_required(VERSION 3.15)
project(sfm_3d_reconstruction)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR C:\\opencv\\mingw-build)
find_package( OpenCV REQUIRED )

add_executable (sfm_3d_reconstruction main.cpp)
target_link_libraries(sfm_3d_reconstruction ${OpenCV_LIBS})