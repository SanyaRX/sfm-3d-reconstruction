//
// Created by Lenovo on 23.10.2019.
//

#include "../include/sfm/StructureFromMotion.h"
#include <iostream>
#include <fstream>

StructureFromMotion::StructureFromMotion(const std::vector<cv::Mat>& images) : images(images)
{
    float parameters[9] = { 2500.0, 0, static_cast<float>(images[0].cols / 2),
                            0, 2500.0, static_cast<float>(images[0].rows / 2),
                            0, 0, 1.0 };
    this->camera_parameters = cv::Mat(3, 3, CV_32F, parameters);


}

StructureFromMotion::StructureFromMotion(const std::string &directory_path, const std::string &list_file_name,
                                         float resize_scale)
{
    this->images = CommonUtilities::loadImages(directory_path, list_file_name, resize_scale);
    float parameters[9] = { 2500.0, 0, static_cast<float>(images[0].cols / 2),
                            0, 2500.0, static_cast<float>(images[0].rows / 2),
                            0, 0, 1.0 };
    this->camera_parameters = cv::Mat(3, 3, CV_32F, parameters);
}

void StructureFromMotion::run()
{

    detectImageFeatures();

    detectImageMatches();

    firstTwoViewsTriangulation();
}

bool StructureFromMotion::detectImageFeatures()
{
    if(images.empty())
        return false;

    for (const cv::Mat& image : this->images)
    {
        Features image_features;
        StereoUtilities::detectFeatures(image, image_features);
        this->images_features.push_back(image_features);
    }
    return true;
}

bool StructureFromMotion::detectImageMatches()
{
    if (images_features.empty())
        return false;

    for (unsigned int i = 0; i < images_features.size() - 1; i++)
    {
        Matches matches;
        StereoUtilities::detectMatches(images_features[i].descriptor, images_features[i + 1].descriptor, matches);
        images_matches.push_back(matches);
    }

    return true;
}

bool StructureFromMotion::firstTwoViewsTriangulation()
{
    if (images_matches.empty())
        return false;
    int i = 0;
    int j = i + 1;
    Features left_features, right_features;
    StereoUtilities::getMatchPoints(images_features[i], images_features[j],
            images_matches[i], left_features, right_features);

    float parameters[9] = { 1500.0, 0, static_cast<float>(images[0].cols / 2),// ..............................
                            0, 1500.0, static_cast<float>(images[0].rows / 2),
                            0, 0, 1.0 };
    this->camera_parameters = cv::Mat(3, 3, CV_32F, parameters);

    float focal = camera_parameters.at<float>(0, 0); //Note: assuming fx = fy
    cv::Point2d pp(camera_parameters.at<float>(0, 2), camera_parameters.at<float>(1, 2));

    cv::Mat mask;
    cv::Mat essential_mat = cv::findEssentialMat(left_features.points2D, right_features.points2D,
            focal, pp, cv::RANSAC, 0.999, 1.0, mask);

    cv::Mat R, t;

    StereoUtilities::decreaseMatrixRank3x3(essential_mat, essential_mat);

    cv::recoverPose(essential_mat, left_features.points2D, right_features.points2D,
            R, t, focal, pp, mask);

    cv::Matx34f pleft = cv::Matx34f::eye();

    cv::Matx34f pright = cv::Matx34f(R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
                     R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
                     R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2));

    this->proj_matrices.push_back(pleft);
    this->proj_matrices.push_back(pright);

    /*Matches prunedMatches;
    for (size_t i = 0; i < mask.rows; i++) {
        if (mask.at<uchar>(i)) {
            prunedMatches.push_back(images_matches[0][i]);
        }
    }
    StereoUtilities::getMatchPoints(images_features[0], images_features[1],
                                    prunedMatches, left_features, right_features);*/
    cv::Mat undistort_left, undistort_right;
    cv::undistortPoints(left_features.points2D, undistort_left, camera_parameters, cv::Mat());
    cv::undistortPoints(right_features.points2D, undistort_right, camera_parameters, cv::Mat());

    cv::Mat points_homogeneous;
    cv::triangulatePoints(pleft, pright, undistort_left, undistort_right, points_homogeneous);

    cv::Mat points3d;
    cv::convertPointsFromHomogeneous(points_homogeneous.t(), points3d);

    cv::Mat rvecLeft;
    Rodrigues(pleft.get_minor<3, 3>(0, 0), rvecLeft);
    cv::Mat tvecLeft(pleft.get_minor<3, 1>(0, 3).t());
    std::cout << rvecLeft << std::endl << tvecLeft << std::endl;

    std::vector<cv::Point2f> projectedOnLeft;
    projectPoints(points3d, rvecLeft, tvecLeft, camera_parameters, cv::Mat(), projectedOnLeft);
    std::vector<cv::Point2f> projectedOnRight;
    projectPoints(points3d, R, t, camera_parameters, cv::Mat(), projectedOnRight);

    for (size_t i = 0; i < points3d.rows; i++) {

        if (norm(projectedOnLeft[i] - left_features.points2D[i]) < 10.0 &&
            norm(projectedOnRight[i] - right_features.points2D[i]) < 10.0) {
            Point3D p;
            p.pt = cv::Point3f(points3d.at<float>(i, 0),
                               points3d.at<float>(i, 1),
                               points3d.at<float>(i, 2)
            );

            //use back reference to point to original features in images
            /*p.originatingViews[imagePair.left]  = leftBackReference [i];
            p.originatingViews[imagePair.right] = rightBackReference[i];
    */
            point_cloud.push_back(p);
        }
    }

    std::ofstream fout("C:\\Users\\Lenovo\\Desktop\\output1.txt");
    for(int i = 0; i < points3d.rows; i++)
    {
        fout << points3d.at<float>(i, 0) << " "
             << points3d.at<float>(i, 1) << " "
             << points3d.at<float>(i, 2) << std::endl;
    }
    CommonUtilities::drawImageMatches(images[i], images[j],
            images_features[i].key_points, images_features[j].key_points, images_matches[i]);
    //cv::projectPoints(points3d);
    return true;
}
