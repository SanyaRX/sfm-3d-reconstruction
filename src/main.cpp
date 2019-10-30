#include <iostream>
#include "../include/sfm/StructureFromMotion.h"

int main() {
    auto images = CommonUtilities::loadImages("D:\\Data Science\\projects\\3d-reconstruction\\mvs-with-opencv\\images\\data",
            "list.txt", 1);
    StructureFromMotion sfm(images);
    sfm.run();

    sfm.savePointCloudXYZ("C:\\Users\\Lenovo\\Desktop\\output_cloud.txt");
    return 0;
}
