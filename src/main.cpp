#include <iostream>
#include "../include/sfm/StructureFromMotion.h"

int main() {
    auto images = CommonUtilities::loadImages("..\\images\\pack",
            "list.txt", 1);
    StructureFromMotion sfm(images);
    sfm.run();

    sfm.savePointCloudXYZ("C:\\Users\\Lenovo\\Desktop\\output_cloud.txt");

    return 0;
}
