//
// Created by Lenovo on 23.10.2019.
//

#include "StereoUtilities.h"

void StereoUtilities::detectKeyPoints(cv::Mat image,  Features &output_features)
{
    cv::Ptr<cv::ORB> detector = cv::ORB::create(MAX_FEATURES);

    detector->detectAndCompute(image, cv::noArray(), output_features.key_points, output_features.descriptor);
}

void StereoUtilities::detectMatches(const cv::Mat &first_descriptors, const cv::Mat &second_descriptors,
        Matches &output_matches)
{
    cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20, 10, 2));
    std::vector<Matches> knn_matches;

    matcher.knnMatch(first_descriptors, second_descriptors, knn_matches, 2);

    for (int k = 0; k < knn_matches.size(); k++) {
        if (knn_matches[k][0].distance < THRES_RATIO * knn_matches[k][1].distance)
            output_matches.push_back(knn_matches[k][0]);
    }
}

void StereoUtilities::getMatchPoints(const Features &left_key_features, const Features &right_key_features,
                                     const Matches &matches,
                                     Features &output_left_features, Features &output_right_features)
{
    output_left_features.key_points.clear();
    output_right_features.key_points.clear();
    output_left_features .descriptor = cv::Mat();
    output_right_features.descriptor = cv::Mat();

    for (cv::DMatch match : matches)
    {
        output_left_features.key_points.push_back(left_key_features.key_points[match.queryIdx]);
        output_right_features.key_points.push_back(right_key_features.key_points[match.trainIdx]);
        output_left_features.descriptor.push_back(left_key_features.descriptor.row(match.queryIdx));
        output_right_features.descriptor.push_back(right_key_features.descriptor.row(match.trainIdx));
    }
}