cmake_minimum_required(VERSION 3.15)
project(sfm_3d_reconstruction)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR C:\\opencv\\mingw-build)
find_package( OpenCV REQUIRED )

add_executable (sfm_3d_reconstruction src/main.cpp src/CommonUtilities.cpp src/CommonUtilities.h src/StructureFromMotion.cpp include/sfm/StructureFromMotion.h src/StereoUtilities.cpp src/StereoUtilities.h)
target_link_libraries(sfm_3d_reconstruction ${OpenCV_LIBS})