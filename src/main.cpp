#include <iostream>
#include "../include/sfm/StructureFromMotion.h"

int main() {
    auto images = CommonUtilities::loadImages("/home/sanyarx/Projects/sfm-3d-reconstructure/images/pack",
            "list.txt", 1);
    StructureFromMotion sfm(images);
    sfm.run();

    sfm.savePointCloudXYZ("output_cloud.txt");

    return 0;
}
