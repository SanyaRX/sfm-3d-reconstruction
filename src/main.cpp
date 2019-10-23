#include <iostream>
#include "../include/sfm/StructureFromMotion.h"

int main() {

    auto images = CommonUtilities::loadImages("D:\\Data Science\\projects\\3d-reconstruction\\mvs-with-opencv\\images\\win-key",
            "list.txt", 0.4);
    int i = 1, j = 2;
    Features first_image_features, second_image_features;
    StereoUtilities::detectKeyPoints(images[i], first_image_features);
    StereoUtilities::detectKeyPoints(images[j], second_image_features);

    Matches first_second_matches;
    StereoUtilities::detectMatches(first_image_features.descriptor, second_image_features.descriptor,
            first_second_matches);

    CommonUtilities::drawImageMatches(images[i], images[j], first_image_features.key_points,
        second_image_features.key_points, first_second_matches);
    return 0;
}
