#include <iostream>
#include "../include/sfm/StructureFromMotion.h"


int main(int argc, char *argv[]) {
    if (argc != 3)
    {
        std::cerr << "Invalid parameters. Use: ./_bin_file_name_  _image_directory_  _output_file_.\n";
        return 0;
    }

    auto images = CommonUtilities::loadImages(argv[1],
            "list.txt", 1);

    if (images.empty())
    {
        std::cerr << "There are no images or no list.txt file in directory " << argv[1] << std::endl;
        return 0;
    }
    else if(images.size() == 1)
    {
        std::cerr << "Required number of images is 2\n";
        return 0;
    }

    StructureFromMotion sfm(images);
    sfm.run();

    sfm.savePointCloudXYZ(argv[2]);

    return 0;
}
