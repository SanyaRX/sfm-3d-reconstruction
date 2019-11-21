cmake_minimum_required(VERSION 3.15)
project(sfm_3d_reconstruction)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

set(PCL_DIR /opt/pcl)





add_executable (sfm_3d_reconstruction src/main.cpp
        src/CommonUtilities.cpp src/CommonUtilities.h
        src/StructureFromMotion.cpp include/sfm/StructureFromMotion.h
        src/StereoUtilities.cpp src/StereoUtilities.h
        src/BundleAdjustment.cpp src/BundleAdjustment.h
        )

target_link_libraries(sfm_3d_reconstruction ${OpenCV_LIBS} ceres )
