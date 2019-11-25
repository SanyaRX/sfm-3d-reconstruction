//
// Created by Lenovo on 23.10.2019.
//

#include "../include/sfm/StructureFromMotion.h"
#include <iostream>
#include <fstream>


StructureFromMotion::StructureFromMotion(const std::vector<cv::Mat>& images) : images(images)
{
    double parameters[9] = { this->focal, 0, static_cast<double>(images[0].cols / 2),
                            0, this->focal, static_cast<double>(images[0].rows / 2),
                            0, 0, 1.0 };
    this->camera_parameters.k_matrix = (cv::Mat_<double>(3,3) <<
            this->focal, 0, static_cast<double>(images[0].cols / 2),
            0, this->focal, static_cast<double>(images[0].rows / 2),
            0, 0, 1.0);
}

void StructureFromMotion::run()
{

    camera_parameters.distortion = cv::Mat_<float>::zeros(1, 4);

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

            Matches proved_matches;
            StereoUtilities::removeOutlierMatches(images_features[i],
                    images_features[j], matches, camera_parameters, proved_matches);

            /*std::cout << "Matches size: " << matches.size() << std::endl
            << "Proved matches size: " << proved_matches.size() << std::endl
            << "---------------------------\n";
            CommonUtilities::drawImageMatches(images[i], images[j], images_features[i].key_points,
                    images_features[j].key_points, matches);*/
            match_matrix[i][j] = proved_matches;

        }
    }
    std::cout << std::endl;
    return true;
}

bool StructureFromMotion::firstTwoViewsTriangulation()
{
    if (match_matrix.empty())
        return false;

    processed_images.resize(images.size(), false);
    processed_pairs.resize(images.size(), std::vector<bool>(images.size(), false));

    std::cout << "Two views triangulation..." << std::endl;

    size_t left_idx = 0, right_idx = 1;
    for (size_t i = 0; i < match_matrix.size(); i++)
    {
        for(size_t j = i + 1; j < match_matrix.size(); j++)
        {
            if(match_matrix[i][j].size() > match_matrix[left_idx][right_idx].size())
            {
                left_idx = i;
                right_idx = j;
            }
        }
    }

    /*CommonUtilities::drawImageMatches(images[left_idx], images[right_idx], images_features[left_idx].key_points,
            images_features[right_idx].key_points, match_matrix[left_idx][right_idx]);*/

    Features left_features, right_features;
    StereoUtilities::getMatchPoints(images_features[left_idx], images_features[right_idx],
                                    match_matrix[left_idx][right_idx], left_features, right_features);


    cv::Point2d pp(camera_parameters.k_matrix.at<double>(0, 2),
            camera_parameters.k_matrix.at<double>(1, 2));

    cv::Mat mask;
    cv::Mat essential_mat = cv::findEssentialMat(left_features.points2D, right_features.points2D,
                                                 this->focal, pp, cv::RANSAC, 0.999, 1.0, mask);

    cv::Mat R, t;

    StereoUtilities::decreaseMatrixRank3x3(essential_mat, essential_mat);

    cv::recoverPose(essential_mat, left_features.points2D, right_features.points2D,
            R, t, this->focal, pp, mask);



    cv::Matx34f pright;

    StereoUtilities::getProjectionMatrixFromRt(R, t, pright);
    this->pose_matrices.resize(images.size());
    this->pose_matrices[left_idx] = cv::Matx34f::eye();
    this->pose_matrices[right_idx] = pright;

    cv::Mat points3d;

    StereoUtilities::triangulatePoints( this->pose_matrices[left_idx], this->pose_matrices[right_idx],
            left_features.points2D, right_features.points2D,
            camera_parameters.k_matrix, points3d);



    std::vector<PointProjection> left_image_track, right_image_track;
    addPointsToPointCloud(left_idx, right_idx, points3d, left_features.points2D, right_features.points2D, match_matrix[left_idx][right_idx],
            left_image_track, right_image_track);

    std::cout << "Point cloud size: " << point_cloud.size() << std::endl;

    /*points_track.push_back(left_image_track); // ???? remove
    points_track.push_back(right_image_track);*/

    BundleAdjustment::processBundleAdjustment(this->point_cloud, this->pose_matrices, this->camera_parameters,
                                              this->images_features);

    processed_images[left_idx] = true;
    processed_images[right_idx] = true;
    processed_pairs[left_idx][right_idx] = true;

    savePointCloudXYZ("../points2views.txt");
    return true;
}

bool StructureFromMotion::addNewViews()
{

    if(pose_matrices.size() < 2)
        return false;

    std::cout << "Adding new views" << std::endl;

    int next_image = 0;
    std::vector<size_t> reconstructed_points, reconstructed_indices;
    while ((next_image = nextImageToReconstruct(reconstructed_points, reconstructed_indices)) != -1)
    {
        processed_images[next_image] = true;

        Points3D existing_points(reconstructed_points.size());
        Points2D current_features(reconstructed_points.size()), points_to_reconstruct_l, points_to_reconstruct_r;
        for (size_t j = 0; j < reconstructed_points.size(); j++)
        {
            existing_points.push_back(point_cloud[reconstructed_points[j]].pt);
            current_features.push_back(images_features[next_image].points2D[reconstructed_indices[j]]);
        }
        if(existing_points.size() < 4){
            std::cout << "Not enough points\n";
            return true;
        }

        std::cout << camera_parameters.k_matrix << std::endl;
        cv::Mat R, t;
        cv::Mat inliers;
        cv::solvePnPRansac(
                existing_points,
                current_features,
                camera_parameters.k_matrix,
                cv::Mat(),
                R,
                t,
                false,
                100,
                10,
                0.99,
                inliers
        );
        double inliersRatio = (double)cv::countNonZero(inliers) / existing_points.size();
        std::cout << "Inliers ratio: " << inliersRatio << std::endl;
        if(inliersRatio < 0.5)
            break;

        cv::Rodrigues(R, R);
        cv::Matx34f pose;
        StereoUtilities::getProjectionMatrixFromRt(R, t, pose);

        pose_matrices[next_image] = pose;

        for (size_t i = 0; i < next_image; i++)
        {
            if(pose_matrices[i](0, 0) || pose_matrices[i](1, 1)
               || pose_matrices[i](2, 2))
            {

                Matches reconstruct_matches;
                for (const cv::DMatch match : match_matrix[i][next_image])
                {
                    if (std::find(reconstructed_indices.begin(), reconstructed_indices.end(), match.trainIdx)
                        == reconstructed_indices.end())
                    {
                        points_to_reconstruct_l.push_back(images_features[i].points2D[match.queryIdx]);
                        points_to_reconstruct_r.push_back(images_features[next_image].points2D[match.trainIdx]);
                        reconstruct_matches.push_back(match);
                    }
                }

                std::cout << "Pair: " << i << " - " << next_image << std::endl;
                std::cout << "Reconstructed points: " << existing_points.size() << std::endl;
                std::cout << "Points to reconstruct: " << points_to_reconstruct_l.size() << std::endl;
                std::cout << "-------------------------\n";

                cv::Mat points3d;

                StereoUtilities::triangulatePoints(pose_matrices[i], pose,
                                                   points_to_reconstruct_l, points_to_reconstruct_r,
                                                   camera_parameters.k_matrix,
                                                   points3d);

                std::vector<PointProjection> left_image_track, right_image_track;
                addPointsToPointCloud(i, next_image, points3d,
                                      points_to_reconstruct_l, points_to_reconstruct_r, reconstruct_matches,
                                      left_image_track, right_image_track);

                std::cout << "Point cloud size: " << point_cloud.size() << std::endl;
                BundleAdjustment::processBundleAdjustment(point_cloud, pose_matrices, camera_parameters,
                                                          images_features);

                std::cout << "After BA: " << point_cloud.size() << std::endl;

                processed_pairs[i][next_image] = true;
                savePointCloudXYZ("../points.txt");
            }
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
    cv::Matx34f pleft = pose_matrices[left_image];
    cv::Matx34f pright = pose_matrices[right_image];

    cv::Mat rvec_left;
    Rodrigues(pleft.get_minor<3, 3>(0, 0), rvec_left);
    cv::Mat tvec_left(pleft.get_minor<3, 1>(0, 3).t());

    cv::Mat rvec_right;
    Rodrigues(pright.get_minor<3, 3>(0, 0), rvec_right);
    cv::Mat tvec_right(pright.get_minor<3, 1>(0, 3).t());

    Points2D projectedOnLeft;
    cv::projectPoints(points3D, rvec_left, tvec_left,  this->camera_parameters.k_matrix, cv::Mat(), projectedOnLeft);

    Points2D projectedOnRight;
    cv::projectPoints(points3D, rvec_right, tvec_right,  this->camera_parameters.k_matrix, cv::Mat(), projectedOnRight);

    for (size_t i = 0; i < points3D.rows; i++)
    {
        if (norm(projectedOnLeft[i] - left_points[i]) < 10.0 &&
            norm(projectedOnRight[i] - right_points[i]) < 10.0) {
            Point3D p;
            p.pt = cv::Point3d(points3D.at<double>(i, 0),
                               points3D.at<double>(i, 1),
                               points3D.at<double>(i, 2)
            );
            p.images.emplace_back(left_image, matches[i].queryIdx);
            p.images.emplace_back(right_image, matches[i].trainIdx);

            bool is_match = false;

            for (size_t j = 0; j < left_image; j++)
            {
                for (const cv::DMatch &match : match_matrix[j][left_image])
                {
                    if(match.trainIdx == matches[i].queryIdx)
                    {
                        p.images.emplace_back(j, match.queryIdx);
                        is_match = true;
                        break;
                    }
                }
            }

            if(!is_match) {
                for (size_t j = left_image + 1; j < images.size(); j++) {
                    if (j != right_image) {
                        for (const cv::DMatch &match : match_matrix[left_image][j]) {
                            if (match.queryIdx == matches[i].queryIdx) {
                                p.images.emplace_back(j, match.trainIdx);
                                break;
                            }
                        }
                    }
                }
            }

            if(p.images[p.images.size() - 1].second < 10000)
                this->point_cloud.push_back(p);

        }
    }

}

int StructureFromMotion::nextImageToReconstruct(std::vector<size_t> &img_reconstructed_pts,
        std::vector<size_t> &img_reconstructed_indices)
{
    std::vector<unsigned int> num_reconstructed_pts(images.size(), 0);
    std::vector<std::vector<size_t>> reconstructed_pts(images.size()), reconstructed_indices(images.size());
    for (size_t i = 0; i < point_cloud.size(); i++)
    {
        for (const std::pair<size_t, size_t> &proj : point_cloud[i].images)
        {
            if(!processed_images[proj.first])
            {
                num_reconstructed_pts[proj.first]++;
                reconstructed_pts[proj.first].push_back(i);
                reconstructed_indices[proj.first].push_back(proj.second);

            }
        }
    }

    int index = 0;

    for (int i = 1; i < num_reconstructed_pts.size(); i++)
    {
        if(num_reconstructed_pts[index] < num_reconstructed_pts[i])
        {
            index = i;
        }
    }

    if(num_reconstructed_pts[index] == 0)
        return -1;

    img_reconstructed_pts = reconstructed_pts[index];
    img_reconstructed_indices = reconstructed_indices[index];

    return index;
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
