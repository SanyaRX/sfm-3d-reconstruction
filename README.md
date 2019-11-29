# Structure from motion
Program for 3d point cloud reconstruction from input image set.


## Building

To build this project you need to install the following libraries:
- OpenCV
- ceres-solver

Follow the next instruction:
1. Make a build directory in project
2. Run cmake command from build directory with path to project as a parmeter
3. Run make command 
4. Run binary file with path to images set directory and output file name as a parameters 
```
mkdir build
cd build
cmake ..
make

```
