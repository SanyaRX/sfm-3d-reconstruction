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
    std::vector<std::vector<PointProjection>> points_track;

    std::vector<bool> processed_images;
    std::vector<std::vector<bool>> processed_pairs;

    PointCloud point_cloud;

    double focal = 1000;
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
     * @return whether the operation has been successful
     */
    bool addNewViews();

    /**
     * Merges new point cloud with existing
     * @param left_image - index of a left image
     * @param right_image - index of a right image
     * @param points3D - new reconstructed points
     * @param left_points - real points3D projections on the left image
     * @param right_points - real points3D projections on the right image
     * @param matches - matches between left_points and right_points
     * @param left_image_track - output array of track points for the left image
     * @param right_image_track - output array of track points for the right image
     */
    void addPointsToPointCloud(int left_image, int right_image, const cv::Mat &points3D,
            const Points2D &left_points, const Points2D &right_points, const Matches &matches,
            std::vector<PointProjection> &left_image_track, std::vector<PointProjection> &right_image_track);

    /**
     * Returns an index of a good fit image to reconstruct points from.
     * @param img_reconstructed_pts - vector of point cloud reconstructed points indices seen by the image
     * @param img_reconstructed_indices - indices of 2d image features according to reconstructed points
     * @return index of the next image
     */
    int nextImageToReconstruct(std::vector<size_t> &img_reconstructed_pts, std::vector<size_t> &img_reconstructed_indices);

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
     * Saves point cloud to txt file in X Y Z format
     * @param file_path - path to save file
     */
    void savePointCloudXYZ(std::string file_path);
};


#endif //SFM_3D_RECONSTRUCTION_STRUCTUREFROMMOTION_H
