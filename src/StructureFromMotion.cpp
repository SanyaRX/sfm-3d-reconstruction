//
// Created by Lenovo on 23.10.2019.
//

#include "../include/sfm/StructureFromMotion.h"
#include <iostream>
#include <fstream>


StructureFromMotion::StructureFromMotion(const std::vector<cv::Mat>& images) : images(images)
{
    this->camera_parameters.k_matrix = (cv::Mat_<double>(3,3) <<
            this->focal, 0, static_cast<double>(images[0].cols / 2),
            0, this->focal, static_cast<double>(images[0].rows / 2),
            0, 0, 1.0);
    this->camera_parameters.distortion = cv::Mat_<double>::zeros(1, 4);
}

void StructureFromMotion::run()
{
    detectImageFeatures();

    detectImageMatches();

    firstTwoViewsTriangulation();

    addNewViews();
}

bool StructureFromMotion::detectImageFeatures()
{
    if(images.empty())
        return false;

    std::cout << "Image features detecting..." << std::endl;

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

    std::cout << "Images matches detecting..." << std::endl;

    match_matrix.resize(images.size(), std::vector<Matches>(images.size()));

    for (size_t i = 0; i < images_features.size() - 1; i++)
    {
        for (size_t j = i + 1; j < images_features.size(); j++)
        {
            std::cout << "\r" << "Pair: " << i << " - " << j << "    " << std::flush;
            Matches matches;
            StereoUtilities::detectMatches(images_features[i].descriptor,
                    images_features[j].descriptor, matches);

            if(matches.size() < 100)
                sorted_inliers_ratio[1.0] = {i, j};
            else
            {
                double ratio = (double) StereoUtilities::findHomographyInliers(images_features[i], images_features[j],
                                                                               matches) / (double) matches.size();
                sorted_inliers_ratio[1 - ratio] = {i, j};
            }
            match_matrix[i][j] = matches;

        }
    }
    std::cout << std::endl;
    return true;
}

bool StructureFromMotion::firstTwoViewsTriangulation()
{
    if (match_matrix.empty())
        return false;

    std::cout << "Two views triangulation..." << std::endl;

    for (const auto &pair : sorted_inliers_ratio) {
        size_t left_idx = pair.second.first, right_idx = pair.second.second;

        /*CommonUtilities::drawImageMatches(images[left_idx], images[right_idx], images_features[left_idx].key_points,
                images_features[right_idx].key_points, match_matrix[left_idx][right_idx]);*/

        cv::Matx34f pright, pleft;
        Matches proved_matches;
        StereoUtilities::findCameraMatricesFromMatch(camera_parameters,
                                                     match_matrix[left_idx][right_idx],
                                                     images_features[left_idx], images_features[right_idx],
                                                     proved_matches, pleft, pright);

        double inliers_ratio = (float)proved_matches.size() / (float)match_matrix[left_idx][right_idx].size();


        if (inliers_ratio < 0.5)
            continue;

        match_matrix[left_idx][right_idx] = proved_matches;

        this->pose_matrices.resize(images.size());
        this->pose_matrices[left_idx] = pleft;
        this->pose_matrices[right_idx] = pright;

        cv::Mat points3d;

        PointCloud pointCloud;
        StereoUtilities::triangulatePoints(this->pose_matrices[left_idx], this->pose_matrices[right_idx],
                                           std::make_pair(left_idx, right_idx), images_features[left_idx],
                                           images_features[right_idx],
                                           match_matrix[left_idx][right_idx], camera_parameters.k_matrix, pointCloud);


        this->point_cloud = pointCloud;

        std::cout << "Point cloud size: " << point_cloud.size() << std::endl;

        BundleAdjustment::processBundleAdjustment(this->point_cloud, this->pose_matrices, this->camera_parameters,
                                                  this->images_features);


        processed_images.insert(left_idx);
        processed_images.insert(right_idx);
        good_images.insert(left_idx);
        good_images.insert(right_idx);

        savePointCloudXYZ("../results/points2views");
        break;
    }
    return true;
}

bool StructureFromMotion::addNewViews()
{

    if(pose_matrices.size() < 2)
        return false;

    std::cout << "Adding new views" << std::endl;

    while (processed_images.size() != images.size())
    {
        Images2D3DMatches matches2D3D = find2D3DMatches();
        if (matches2D3D.empty())
            break;

        size_t best_image;
        size_t best_num_matches = 0;
        for (const auto &match2D3D : matches2D3D)
        {
            const size_t num_matches = match2D3D.second.points2D.size();
            if (num_matches > best_num_matches)
            {
                best_image = match2D3D.first;
                best_num_matches = num_matches;
            }
        }

        processed_images.insert(best_image);

        cv::Mat R, t;
        cv::Mat inliers;
        if (matches2D3D[best_image].points3D.size() < 6)
        {
            std::cerr << "Not enough points for solvePnP\n";
            continue;
        }

        cv::solvePnPRansac(
                matches2D3D[best_image].points3D,
                matches2D3D[best_image].points2D,
                camera_parameters.k_matrix,
                camera_parameters.distortion,
                R,
                t,
                false,
                100,
                10,
                0.99,
                inliers
        );
        double inliers_ratio = (double)cv::countNonZero(inliers) / matches2D3D[best_image].points3D.size();
        std::cout << "Inliers ratio: " << inliers_ratio << std::endl;
        if(inliers_ratio < 0.5)
            continue;
        cv::Mat rotate_matrix;
        cv::Rodrigues(R, rotate_matrix);
        cv::Matx34f pose;
        StereoUtilities::getProjectionMatrixFromRt(rotate_matrix, t, pose);

        pose_matrices[best_image] = pose;
        bool success_push = false;
        for (const int good_image : good_images) {

            size_t left_idx = (good_image < best_image) ? good_image : best_image;
            size_t right_idx = (good_image < best_image) ? best_image : good_image;

            cv::Matx34f pright = cv::Matx34f::eye(), pleft = cv::Matx34f::eye();
            Matches proved_matches;
            bool is_success = StereoUtilities::findCameraMatricesFromMatch(camera_parameters,
                                                         match_matrix[left_idx][right_idx],
                                                         images_features[left_idx],
                                                         images_features[right_idx],
                                                         proved_matches, pleft, pright);
            if(!is_success)
                continue;
            match_matrix[left_idx][right_idx] = proved_matches;

            PointCloud cloud;
            StereoUtilities::triangulatePoints(
                    pose_matrices[left_idx],
                    pose_matrices[right_idx],
                    {left_idx, right_idx},
                    images_features[left_idx],
                    images_features[right_idx],
                    match_matrix[left_idx][right_idx],
                    camera_parameters.k_matrix,
                    cloud
            );


            addPointsToPointCloud(cloud);
            success_push = true;

        }
        if (success_push)
            BundleAdjustment::processBundleAdjustment(this->point_cloud, this->pose_matrices, this->camera_parameters,
                                                  this->images_features);

        good_images.insert(best_image);
        savePointCloudXYZ("../results/prom");
    }
    return true;
}

void StructureFromMotion::addPointsToPointCloud(const PointCloud &cloud)
{
    for (const Point3D& p : cloud)
    {
        const cv::Point3d newPoint = p.pt;

        bool found_existing_views = false;
        bool found_matching_point = false;
        for (Point3D& existing_point : this->point_cloud)
        {
            if (norm(existing_point.pt - newPoint) < 0.01)
            {
                found_matching_point = true;

                for (const auto& new_kv : p.images)
                {
                    for (const auto& existing_kv : existing_point.images)
                    {
                        bool found_matching_feature = false;

                        const bool in_new_left = new_kv.first < existing_kv.first;
                        const int left_idx = (in_new_left) ? new_kv.first : existing_kv.first;
                        const int left_feature_idx  = (in_new_left) ? new_kv.second : existing_kv.second;
                        const int right_idx = (in_new_left) ? existing_kv.first : new_kv.first;
                        const int right_feature_idx = (in_new_left) ? existing_kv.second : new_kv.second;

                        const Matches& matching = match_matrix[left_idx][right_idx];
                        for (const cv::DMatch& match : matching)
                        {
                            if (match.queryIdx == left_feature_idx
                                    && match.trainIdx == right_feature_idx
                                    && match.distance < 10.0)
                            {

                                found_matching_feature = true;
                                break;
                            }
                        }

                        if (found_matching_feature)
                        {
                            existing_point.images[new_kv.first] = new_kv.second;
                            found_existing_views = true;
                        }
                    }
                }
            }
            if (found_existing_views)
            {
                break;
            }
        }

        if (!found_existing_views && !found_matching_point)
        {
            point_cloud.push_back(p);
        }
    }


}

StructureFromMotion::Images2D3DMatches StructureFromMotion::find2D3DMatches()
{
    Images2D3DMatches matches;

    for (size_t image_idx = 0; image_idx < images.size(); image_idx++)
    {
        if (processed_images.find(image_idx) != processed_images.end())
            continue;


        Image2D3DMatch match2D3D;

        for (const Point3D& cloud_point : point_cloud)
        {
            bool found_point = false;

            for (const auto& orig_view_point : cloud_point.images)
            {
                const int orig_view_index = orig_view_point.first;
                const int orig_view_feature_idx = orig_view_point.second;

                const int left_idx  = (orig_view_index < image_idx) ? orig_view_index : image_idx;
                const int right_idx = (orig_view_index < image_idx) ? image_idx : orig_view_index;

                for (const cv::DMatch& m : match_matrix[left_idx][right_idx])
                {
                    int matched_point_new_view = -1;
                    if (orig_view_index < image_idx)
                    {
                        if (m.queryIdx == orig_view_feature_idx)
                        {
                            matched_point_new_view = m.trainIdx;
                        }
                    }
                    else
                    {
                        if (m.trainIdx == orig_view_feature_idx)
                        {
                            matched_point_new_view = m.queryIdx;
                        }
                    }
                    if (matched_point_new_view >= 0)
                    {
                        const Features& new_view_features = images_features[image_idx];
                        match2D3D.points2D.push_back(new_view_features.points2D[matched_point_new_view]);
                        match2D3D.points3D.push_back(cloud_point.pt);
                        found_point = true;
                        break;
                    }
                }

                if (found_point)
                    break;

            }
        }

        matches[image_idx] = match2D3D;
    }

    return matches;
}
void StructureFromMotion::savePointCloudXYZ(std::string file_path)
{
    std::ofstream fout(file_path + ".xyz");

    for(const auto &point : point_cloud)
    {
        fout << point.pt.x << " "
             << point.pt.y << " "
             << point.pt.z << std::endl;
    }

}
