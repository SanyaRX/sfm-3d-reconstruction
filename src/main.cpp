#include <iostream>
#include "../include/sfm/StructureFromMotion.h"



int main() {
    auto images = CommonUtilities::loadImages("../images/win-key",
            "list.txt", 0.4);
    StructureFromMotion sfm(images);
    sfm.run();

    sfm.savePointCloudXYZ("../points.txt");

    return 0;
}
