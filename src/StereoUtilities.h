//
// Created by Lenovo on 23.10.2019.
//

#ifndef SFM_3D_RECONSTRUCTION_STEREOUTILITIES_H
#define SFM_3D_RECONSTRUCTION_STEREOUTILITIES_H

#import "CommonUtilities.h"

class StereoUtilities {
    constexpr const static float THRES_RATIO = 0.7f;
    constexpr const static unsigned int MAX_FEATURES = 8000;

public:

    /**
     * Detects image features.
     * @param image - image to work with
     * @param output_features - image features for output
     */
    static void detectFeatures(cv::Mat image,
                               Features &output_features);

    /**
     * Detects feature matches between two images.
     * @param first_descriptors - first image features descriptor
     * @param second_descriptors - second image features descriptor
     * @param output_matches - output vector for matches
     */
    static void detectMatches(const cv::Mat &first_descriptor,
                              const cv::Mat &second_descriptor,
                              Matches &output_matches);

    /**
     * Gets matched features on right and left images.
     * @param left_key_features- left image features
     * @param right_keyfeatures - right image features
     * @param matches - key points matches
     * @param left_points - output array for matched features on the left image
     * @param right_points - output array for matched features on the right image
     * @param left_image_proj - output array for matched feature indexes on the left image
     * @param right_image_proj - output array for matched feature indexes on the right image
     */
    static void getMatchPoints(const Features &left_key_features,
                               const Features &right_key_features,
                               const Matches &matches,
                               Features &output_left_features,
                               Features &output_right_features,
                               std::vector<int> &left_image_proj,
                               std::vector<int> &right_image_proj);

    /**
     * Decreases 3x3 matrix's rank from 3 to 2.
     * @param matrix - matrix
     * @param output_matrix - output matrix with decreased rank
     * @return whether operation has been successful
     */
    static bool decreaseMatrixRank3x3(const cv::Mat &matrix, cv::Mat &output_matrix);

    /**
     * Converts rotation and translation matrices to projection matrix
     * @param R - rotation matrix
     * @param t - translation matrix
     * @param output_matrix - output projection matrix
     */
    static void getProjectionMatrixFromRt(const cv::Mat &R, const cv::Mat &t, cv::Matx34f &output_matrix);

    /**
     * Reconstructs 3D point cloud using points on two images.
     * @param pleft - left image projection matrix
     * @param pright - right image projection matrix
     * @param image_pair - pair of images to process
     * @param left_features - left image match points
     * @param right_features - right image key points
     * @param matches - images points matches
     * @param camera_parameters - camera parameters
     * @param output_points - output array of reconstructed 3D points
     */
    static void triangulatePoints(const cv::Matx34f &pleft, const cv::Matx34f &pright, const std::pair<int, int> &image_pair,
            const Features &left_features, const Features &right_features, const Matches& matches, const cv::Mat &camera_parameters,
            PointCloud& output_points);


    /**
     * Recover relative camera poses from matches
     * @param camera_parameters - camera parameters
     * @param matches - matches between two views
     * @param features_left - left view features
     * @param features_right - right view features
     * @param pruned_matches - output proved matches
     * @param pleft - output left camera matrix
     * @param pright - output right camera matrix
     * @return
     */
    static bool findCameraMatricesFromMatch(const CameraParameters& camera_parameters, const Matches& matches,
            const Features& features_left, const Features& features_right,
            Matches& pruned_matches, cv::Matx34f& pleft,cv::Matx34f& pright);
    /**
     * Removes outliers from matches using RANSAC-based robust method.
     * @param left_image_features - left image features
     * @param right_im1age_features - right image features
     * @param matches - image matches
     * @param proved_matches - output array for proved matches
     */
    static void removeOutlierMatches(const Features &left_image_features,
                                     const Features &right_im1age_features,
                                     const Matches &matches, const CameraParameters &camera_parameters,
                                     Matches &proved_matches);
    /**
     * Returns number of homography inliers for two images.
     * @param left - left image features
     * @param right l right image features
     * @param matches - images matches
     * @return - number of homography inliers
     */
    static int findHomographyInliers(const Features& left, const Features& right, const Matches& matches);
};



#endif //SFM_3D_RECONSTRUCTION_STEREOUTILITIES_H
