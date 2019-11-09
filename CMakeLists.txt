cmake_minimum_required(VERSION 3.15)
project(sfm_3d_reconstruction)

set(CMAKE_CXX_STANDARD 17)

#set(OpenCV_DIR C:\\opencv\\mingw-build)
find_package( OpenCV REQUIRED )
cmake_minimum_required( VERSION 2.8 )

#set(PCL_ROOT C:\\Program Files\\PCL 1.9.1\\share\\doc\\pcl-1.9\\tutorials\\sources\\build)
#find_package( PCL 1.9 REQUIRED )

#include_directories( ${PCL_INCLUDE_DIRS} )
#add_definitions( ${PCL_DEFINITIONS} )

#link_directories( ${PCL_LIBRARY_DIRS} )

add_executable (sfm_3d_reconstruction src/main.cpp
        src/CommonUtilities.cpp src/CommonUtilities.h
        src/StructureFromMotion.cpp include/sfm/StructureFromMotion.h
        src/StereoUtilities.cpp src/StereoUtilities.h)

target_link_libraries(sfm_3d_reconstruction ${OpenCV_LIBS})