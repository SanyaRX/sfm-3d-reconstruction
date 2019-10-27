#include <iostream>
#include "../include/sfm/StructureFromMotion.h"

int main() {
    auto images = CommonUtilities::loadImages("D:\\Data Science\\projects\\3d-reconstruction\\mvs-with-opencv\\images\\watch",
            "list.txt", 0.4);
    StructureFromMotion sfm(images);
    sfm.run();
    return 0;
}
