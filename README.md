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
./sfm-3d-reconstruction _images_directory_path_ _ouput_file_name_
```
### Note:
Image directory must contain a list.txt. In list.txt file you need to list images names that you want to use in reconstruction.
## Example
To run example build project as mentioned before and run the following line:
```
./sfm_3d_reconstruction ../images/stone/ ../results/stone
```
To visualize results you can use MeshLab program. Just drag and drop result file to MeshLab window.
