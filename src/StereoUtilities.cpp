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

    std::vector<Matches> knn_matches;
    //cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20, 10, 2));
    //matcher.knnMatch(first_descriptors, second_descriptors, knn_matches, 2);
    auto matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->knnMatch(first_descriptors, second_descriptors, knn_matches, 2);


    for (Matches &knn_match : knn_matches) {
        if (knn_match[0].distance < THRES_RATIO * knn_match[1].distance)
            output_matches.push_back(knn_match[0]);
    }
}

void StereoUtilities::getMatchPoints(const Features &left_key_features, const Features &right_key_features,
                                     const Matches &matches,Features &output_left_features,
                                     Features &output_right_features,
                                     std::vector<int> &left_image_proj,
                                     std::vector<int> &right_image_proj)
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
        left_image_proj.push_back(match.queryIdx);
        right_image_proj.push_back(match.trainIdx);
    }
}

bool StereoUtilities::decreaseMatrixRank3x3(const cv::Mat &matrix, cv::Mat &output_matrix)
{
    if(matrix.cols != 3 || matrix.rows != 3)
        return false;

    cv::Mat w(3, 3, CV_64F), u(3, 3, CV_64F), vt(3, 3, CV_64F);
    cv::SVDecomp(matrix, w, u, vt);

    output_matrix.reshape(3, 3);
    output_matrix.convertTo(output_matrix, CV_64F);

    w.row(0).setTo(1);
    w.row(1).setTo(1);
    w.row(2).setTo(0);

    output_matrix = (u * cv::Mat::diag(w)) * vt;
    return true;
}

void StereoUtilities::getProjectionMatrixFromRt(const cv::Mat &R, const cv::Mat &t, cv::Matx34f &output_matrix)
{
    output_matrix = cv::Matx34f(R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
                R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
                R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2));
}

void StereoUtilities::triangulatePoints(const cv::Matx34f &pleft, const cv::Matx34f &pright,
                                        const std::pair<int, int> &image_pair,
                                        const Features &left_features, const Features &right_features,
                                        const Matches& matches,
                                        const cv::Mat &camera_parameters, PointCloud &output_points)
{
    std::vector<int> leftBackReference;
    std::vector<int> rightBackReference;
    Features alignedLeft;
    Features alignedRight;
    getMatchPoints(
            left_features,
            right_features,
            matches,
            alignedLeft,
            alignedRight,
            leftBackReference,
            rightBackReference);

    Points2D left_points = alignedLeft.points2D;
    Points2D right_points = alignedRight.points2D;

    Points2D undistort_left, undistort_right;
    cv::undistortPoints(left_points, undistort_left, camera_parameters, cv::Mat());
    cv::undistortPoints(right_points, undistort_right, camera_parameters, cv::Mat());

    cv::Mat points_homogeneous;
    cv::triangulatePoints(pleft, pright, undistort_left, undistort_right, points_homogeneous);

    cv::Mat points3d;
    cv::convertPointsFromHomogeneous(points_homogeneous.t(), points3d);

    cv::Mat rvec_left;
    cv::Rodrigues(pleft.get_minor<3, 3>(0, 0), rvec_left);
    cv::Mat tvec_left(pleft.get_minor<3, 1>(0, 3).t());

    cv::Mat rvec_right;
    cv::Rodrigues(pright.get_minor<3, 3>(0, 0), rvec_right);
    cv::Mat tvec_right(pright.get_minor<3, 1>(0, 3).t());

    Points2D projectedOnLeft;
    cv::projectPoints(points3d, rvec_left, tvec_left,  camera_parameters, cv::Mat(), projectedOnLeft);

    Points2D projectedOnRight;
    cv::projectPoints(points3d, rvec_right, tvec_right,  camera_parameters, cv::Mat(), projectedOnRight);

    for (size_t i = 0; i < points3d.rows; i++)
    {
        if (norm(projectedOnLeft[i] - left_points[i]) < 10.0 &&
            norm(projectedOnRight[i] - right_points[i]) < 10.0) {
            Point3D p;
            p.pt = cv::Point3d(points3d.at<double>(i, 0),
                               points3d.at<double>(i, 1),
                               points3d.at<double>(i, 2)
            );
            p.images[image_pair.first] = leftBackReference [i];
            p.images[image_pair.second] = rightBackReference[i];

            output_points.push_back(p);
        }
    }
}

void StereoUtilities::removeOutlierMatches(const Features &left_image_features, const Features &right_im1age_features,
                                           const Matches &matches, const CameraParameters &camera_parameters,
                                           Matches &proved_matches)
{
    proved_matches.clear();

    if(matches.size() < 8)
    {
        proved_matches = matches;
        return;
    }

    Features left_match_points, right_match_points;
    std::vector<int> left_image_proj, right_image_proj;
    getMatchPoints(left_image_features, right_im1age_features, matches, left_match_points, right_match_points,
                   left_image_proj, right_image_proj);

    double focal = camera_parameters.k_matrix.at<double>(0, 0);
    cv::Point2d pp(camera_parameters.k_matrix.at<double>(0, 2),
            camera_parameters.k_matrix.at<double>(1, 2));

    cv::Mat E, R, t;
    cv::Mat mask;
    E = findEssentialMat(left_match_points.points2D, right_match_points.points2D, focal, pp, cv::RANSAC,
            0.999, 1.0, mask);

    recoverPose(E, left_match_points.points2D, right_match_points.points2D, R, t, focal, pp, mask);

    for (size_t i = 0; i < mask.rows; i++)
    {
        if((int)mask.at<uchar>(i, 0))
            proved_matches.push_back(matches[i]);
    }
}

bool StereoUtilities::findCameraMatricesFromMatch(const CameraParameters& camera_parameters, const Matches& matches,
        const Features& features_left, const Features& features_right,
        Matches& pruned_matches, cv::Matx34f& pleft,cv::Matx34f& pright) {

    if (matches.size() < 100)
        return false;

    double focal = camera_parameters.k_matrix.at<double>(0, 0);
    cv::Point2d pp(camera_parameters.k_matrix.at<double>(0, 2), camera_parameters.k_matrix.at<double>(1, 2));

    Features left_features;
    Features right_features;
    std::vector<int> left, right;
    getMatchPoints(features_left, features_right, matches, left_features, right_features, left, right);


    cv::Mat E, R, t;
    cv::Mat mask;
    E = findEssentialMat(left_features.points2D, right_features.points2D, focal, pp, cv::RANSAC,
            0.999, 1.0, mask);


    cv::recoverPose(E, left_features.points2D, right_features.points2D, R, t, focal, pp, mask);

    pleft = cv::Matx34f::eye();
    getProjectionMatrixFromRt(R, t, pright);

    pruned_matches.clear();
    for (size_t i = 0; i < mask.rows; i++) {
        if (mask.at<uchar>(i)) {
            pruned_matches.push_back(matches[i]);
        }
    }

    return true;
}
int StereoUtilities::findHomographyInliers(
        const Features& left,
        const Features& right,
        const Matches& matches) {

    Features left_features;
    Features right_features;
    std::vector<int> left_image_proj, right_image_proj;
    getMatchPoints(left, right, matches, left_features, right_features, left_image_proj, right_image_proj);

    cv::Mat inlierMask;
    cv::Mat homography;
    if(matches.size() >= 4) {
        homography = findHomography(left_features.points2D, right_features.points2D,
                                    cv::RANSAC, 10, inlierMask);
    }

    if(matches.size() < 4 || homography.empty()) {
        return 0;
    }

    return countNonZero(inlierMask);
}