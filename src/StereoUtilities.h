//
// Created by Lenovo on 23.10.2019.
//

#ifndef SFM_3D_RECONSTRUCTION_STEREOUTILITIES_H
#define SFM_3D_RECONSTRUCTION_STEREOUTILITIES_H

#import "CommonUtilities.h"

class StereoUtilities {
    constexpr const static float THRES_RATIO = 0.7f;
    constexpr const static unsigned int MAX_FEATURES = 800;

public:

    /**
     * Detects image features
     * @param image - image to work with
     * @param output_features - image features for output
     */
    static void detectFeatures(cv::Mat image,
                               Features &output_features);

    /**
     * Detects feature matches between two images
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
     */
    static void getMatchPoints(const Features &left_key_features,
                               const Features &right_key_features,
                               const Matches &matches,
                               Features &output_left_features,
                               Features &output_right_features);

    /**
     * Decreases 3x3 matrix's rank from 3 to 2
     * @param matrix - matrix
     * @param output_matrix - output matrix with decreased rank
     * @return whether operation has been successful
     */
    static bool decreaseMatrixRank3x3(const cv::Mat &matrix, cv::Mat &output_matrix);
};


#endif //SFM_3D_RECONSTRUCTION_STEREOUTILITIES_H
