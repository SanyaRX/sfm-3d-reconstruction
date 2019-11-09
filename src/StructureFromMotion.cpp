//
// Created by Lenovo on 23.10.2019.
//

#include "../include/sfm/StructureFromMotion.h"
#include <iostream>
#include <fstream>


StructureFromMotion::StructureFromMotion(const std::vector<cv::Mat>& images) : images(images)
{
    float parameters[9] = { 1500.0, 0, static_cast<float>(images[0].cols / 2),
                            0, 1500.0, static_cast<float>(images[0].rows / 2),
                            0, 0, 1.0 };
    this->camera_parameters = cv::Mat(3, 3, CV_32F, parameters);


}

StructureFromMotion::StructureFromMotion(const std::string &directory_path, const std::string &list_file_name,
                                         float resize_scale)
{
    this->images = CommonUtilities::loadImages(directory_path, list_file_name, resize_scale);
    float parameters[9] = { 1500.0, 0, static_cast<float>(images[0].cols / 2),
                            0, 1500.0, static_cast<float>(images[0].rows / 2),
                            0, 0, 1.0 };
    this->camera_parameters = cv::Mat(3, 3, CV_32F, parameters);
}

void StructureFromMotion::run()
{
    detectImageFeatures();

    detectImageMatches();
    CommonUtilities::drawImageMatches(images[0], images[1], images_features[0].key_points, images_features[1].key_points,
            images_matches[0]);
    firstTwoViewsTriangulation();

    addNewViews();

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

    float parameters[9] = { 1500.0, 0, static_cast<float>(images[0].cols / 2),
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

    cv::Matx34f pright;
    StereoUtilities::getProjectionMatrixFromRt(R, t, pright);

    this->proj_matrices.push_back(pleft);
    this->proj_matrices.push_back(pright);

    Matches pruned_matches;
    for (size_t i = 0; i < mask.rows; i++) {
        if (mask.at<uchar>(i)) {
            pruned_matches.push_back(images_matches[0][i]);
        }
    }
    StereoUtilities::getMatchPoints(images_features[0], images_features[1],
                                    pruned_matches, left_features, right_features);

    cv::Mat points3d;
    StereoUtilities::triangulatePoints(pleft, pright, left_features.points2D, right_features.points2D,
            camera_parameters, points3d);

    std::vector<PointProjection> left_image_track, right_image_track;

    addPointsToPointCloud(i, j, points3d, left_features.points2D, right_features.points2D, pruned_matches,
            left_image_track, right_image_track);

    points_track.push_back(left_image_track);
    points_track.push_back(right_image_track);

    return true;
}

bool StructureFromMotion::addNewViews()
{
    if(proj_matrices.size() < 2)
        return false;

    for (int i = 2; i < images.size(); i++)
    {
        Points3D existing_points;
        Points2D current_features, points_to_reconstruct_l, points_to_reconstruct_r;
        Matches reconstruct_matches;
        for (const cv::DMatch &match : images_matches[i - 1])
        {
            int idx = StereoUtilities::isPointReconstructed(points_track[i - 1], match.queryIdx);

            if (idx != -1)
            {
                existing_points.push_back(points_track[i - 1][idx].pt);
                current_features.push_back(images_features[i].points2D[match.trainIdx]);
            }
            else
            {
                points_to_reconstruct_l.push_back(images_features[i - 1].points2D[match.queryIdx]);
                points_to_reconstruct_r.push_back(images_features[i].points2D[match.trainIdx]);
                reconstruct_matches.push_back(match);
            }
        }

        cv::Mat camera_params;
        float parameters[9] = { 1500.0, 0, static_cast<float>(images[0].cols / 2),
                                0, 1500.0, static_cast<float>(images[0].rows / 2),
                                0, 0, 1.0 };
        camera_params = cv::Mat(3, 3, CV_32F, parameters);
        std::cout << current_features.size() << std::endl;
        if(existing_points.size() >= 7)
        {
            cv::Mat R, t;
            cv::solvePnPRansac(existing_points, current_features, camera_params, cv::Mat(), R, t);

            cv::Matx34f projection_matrix;
            cv::Rodrigues(R, R);

            StereoUtilities::getProjectionMatrixFromRt(R, t, projection_matrix);
            proj_matrices.push_back(projection_matrix);

            cv::Mat points3d;

            StereoUtilities::triangulatePoints(proj_matrices[i - 1], projection_matrix,
                                               points_to_reconstruct_l, points_to_reconstruct_r, camera_params,
                                               points3d);

            std::vector<PointProjection> left_image_track, right_image_track;
            addPointsToPointCloud(i - 1, i, points3d,
                    points_to_reconstruct_l, points_to_reconstruct_r, reconstruct_matches,
                    left_image_track, right_image_track);

            points_track.push_back(left_image_track);
            points_track[i - 1].insert(points_track[i - 1].end(), right_image_track.begin(),
                    right_image_track.end());
        }
    }
    return true;
}

void StructureFromMotion::addPointsToPointCloud(int left_image, int right_image,
                                                const cv::Mat &points3D, const Points2D &left_points,
                                                const Points2D &right_points, const Matches &matches,
                                                std::vector<PointProjection> &left_image_track,
                                                std::vector<PointProjection> &right_image_track)
{
    cv::Matx34f pleft = proj_matrices[left_image];
    cv::Matx34f pright = proj_matrices[right_image];

    cv::Mat rvec_left;
    Rodrigues(pleft.get_minor<3, 3>(0, 0), rvec_left);
    cv::Mat tvec_left(pleft.get_minor<3, 1>(0, 3).t());

    cv::Mat rvec_right;
    Rodrigues(pright.get_minor<3, 3>(0, 0), rvec_right);
    cv::Mat tvec_right(pright.get_minor<3, 1>(0, 3).t());

    cv::Mat camera_params;
    float parameters[9] = { 1500.0, 0, static_cast<float>(images[0].cols / 2),
                            0, 1500.0, static_cast<float>(images[0].rows / 2),
                            0, 0, 1.0 };

    camera_params = cv::Mat(3, 3, CV_32F, parameters);

    Points2D projectedOnLeft;
    projectPoints(points3D, rvec_left, tvec_left, camera_params, cv::Mat(), projectedOnLeft);

    Points2D projectedOnRight;
    projectPoints(points3D, rvec_right, tvec_right, camera_params, cv::Mat(), projectedOnRight);

    for (int i = 0; i < points3D.rows; i++)
    {
        //std::cout << norm(projectedOnLeft[i] - left_points[i]) << std::endl;
        /*if (norm(projectedOnLeft[i] - left_points[i]) < 10.0 &&
            norm(projectedOnRight[i] - right_points[i]) < 10.0)*/ {
            Point3D p;
            p.pt = cv::Point3f(points3D.at<float>(i, 0),
                               points3D.at<float>(i, 1),
                               points3D.at<float>(i, 2)
            );
            p.images.emplace_back(right_image, matches[i].trainIdx);
            this->point_cloud.push_back(p);
            left_image_track.push_back({p.pt, matches[i].queryIdx});
            right_image_track.push_back({p.pt, matches[i].trainIdx});
        }
    }

}

void StructureFromMotion::savePointCloudXYZ(std::string file_path)
{
    std::ofstream fout(file_path);

    for(const auto &point : point_cloud)
    {
        fout << point.pt.x << " "
             << -point.pt.y << " "
             << point.pt.z << std::endl;
    }

}
