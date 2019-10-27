//
// Created by Lenovo on 23.10.2019.
//

#include "StereoUtilities.h"

void StereoUtilities::detectFeatures(cv::Mat image, Features &output_features)
{
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(MAX_FEATURES);

    detector->detectAndCompute(image, cv::noArray(), output_features.key_points, output_features.descriptor);
    for (const cv::KeyPoint &key_point : output_features.key_points)
    {
        output_features.points2D.push_back(key_point.pt);
    }
}

void StereoUtilities::detectMatches(const cv::Mat &first_descriptors, const cv::Mat &second_descriptors,
        Matches &output_matches)
{
    cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20, 10, 2));
    std::vector<Matches> knn_matches;

    matcher.knnMatch(first_descriptors, second_descriptors, knn_matches, 2);

    for (Matches &knn_match : knn_matches) {
        if (knn_match[0].distance < THRES_RATIO * knn_match[1].distance)
            output_matches.push_back(knn_match[0]);
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
    output_left_features.points2D.clear();
    output_right_features.points2D.clear();
    for (cv::DMatch match : matches)
    {
        output_left_features.key_points.push_back(left_key_features.key_points[match.queryIdx]);
        output_right_features.key_points.push_back(right_key_features.key_points[match.trainIdx]);
        output_left_features.descriptor.push_back(left_key_features.descriptor.row(match.queryIdx));
        output_right_features.descriptor.push_back(right_key_features.descriptor.row(match.trainIdx));
        output_left_features.points2D.push_back(left_key_features.points2D[match.queryIdx]);
        output_right_features.points2D.push_back(right_key_features.points2D[match.trainIdx]);
    }
}

bool StereoUtilities::decreaseMatrixRank3x3(const cv::Mat &matrix, cv::Mat &output_matrix)
{
    if(matrix.cols != 3 || matrix.rows != 3)
        return false;

    cv::Mat w(3, 3, CV_32F), u(3, 3, CV_32F), vt(3, 3, CV_32F);
    cv::SVDecomp(matrix, w, u, vt);

    output_matrix.reshape(3, 3);
    output_matrix.convertTo(output_matrix, CV_32F);

    w.row(2).setTo(0);

    output_matrix = (u * cv::Mat::diag(w)) * vt;
    return true;
}