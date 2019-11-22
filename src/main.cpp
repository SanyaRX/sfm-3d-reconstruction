#include <iostream>
#include "../include/sfm/StructureFromMotion.h"


int main() {
    auto images = CommonUtilities::loadImages("../images/stone",
            "list.txt", 1);
    StructureFromMotion sfm(images);
    sfm.run();

    sfm.savePointCloudXYZ("../points.txt");

    return 0;
}
