//
// Created by Lenovo on 23.10.2019.
//

#ifndef SFM_3D_RECONSTRUCTION_STRUCTUREFROMMOTION_H
#define SFM_3D_RECONSTRUCTION_STRUCTUREFROMMOTION_H

#include "../../src/CommonUtilities.h"
#include "../../src/StereoUtilities.h"
#include "../../src/BundleAdjustment.h"
#include <algorithm>

class StructureFromMotion {

    std::vector<cv::Mat> images;
    std::vector<Features> images_features;
    std::vector<std::vector<Matches>> match_matrix;
    CameraParameters camera_parameters;
    std::vector<cv::Matx34d> pose_matrices;

    std::set<int> processed_images;
    std::set<int> good_images;


    PointCloud point_cloud;

    double focal = 1000;

    typedef std::map<int, Image2D3DMatch> Images2D3DMatches;
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
     * Computes projection matrices for first two views and triangulates points seen by this views
     * @return whether the operation has been successful
     */
    bool firstTwoViewsTriangulation();

    /**
     * Processes the remaining views.
     * @param pointCloud - new point cloud to merge with existing one
     * @return whether the operation has been successful
     */
    bool addNewViews();

    /**
     * Merges new point cloud with existing one.
     * @param cloud - new point cloud
     */
    void addPointsToPointCloud(const PointCloud &cloud);

    /**
     * Returns 2d points of some image with maximum corresponding reconstructed 3d points.
     */
    Images2D3DMatches find2D3DMatches();

public:

    /**
     * Constructor that takes images
     * @param images - images of an object
     */
    StructureFromMotion(const std::vector<cv::Mat>& images);

    /**
     * Runs Structure From Motion algorithm
     */
    void run();

    /**
     * Saves point cloud to .xyz file in X Y Z format
     * @param file_path - path to save file
     */
    void savePointCloudXYZ(std::string file_path);
};


#endif //SFM_3D_RECONSTRUCTION_STRUCTUREFROMMOTION_H
