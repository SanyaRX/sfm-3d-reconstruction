//
// Created by Lenovo on 23.10.2019.
//

#ifndef SFM_3D_RECONSTRUCTION_STRUCTUREFROMMOTION_H
#define SFM_3D_RECONSTRUCTION_STRUCTUREFROMMOTION_H

#include "../../src/CommonUtilities.h"
#include "../../src/StereoUtilities.h"

class StructureFromMotion {
    cv::Mat camera_parameters;
    std::vector<cv::Mat> images;
    std::vector<Features> images_features;
    std::vector<Matches> images_matches;
    std::vector<cv::Matx34f> proj_matrices;
    PointCloud point_cloud;
    /**
     * Detects features of all the images
     * @return whether the operation has been successful
     */
    bool detectImageFeatures();

    /**
     * Detects matches between image features
     * @return whether the operation has been successful
     */
    bool detectImageMatches();

    /**
     * Compute projection matrices for first two views and triangulate points seen by this views
     * @return whether the operation has been successful
     */
    bool firstTwoViewsTriangulation();
public:
    /**
     * Constructor that load images of an object from directory
     * @param directory_path - path to directory with images
     * @param list_file_name - name of file with image names list
     * @param resize_scale - float in interval (0, 1) defines resized image sizes. If out of (0, 1) then image won't be resized.
     */
    StructureFromMotion(const std::string &directory_path,
                        const std::string &list_file_name,
                        float resize_scale = 0);

    /**
     * Constructor that takes images
     * @param images - images of an object
     */
    StructureFromMotion(const std::vector<cv::Mat>& images);

    /**
     * Runs Structure From Motion algorithm
     */
    void run();
};


#endif //SFM_3D_RECONSTRUCTION_STRUCTUREFROMMOTION_H
