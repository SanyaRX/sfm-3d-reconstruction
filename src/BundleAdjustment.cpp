//
// Created by Lenovo on 11/8/2019.
//

#include "BundleAdjustment.h"

void initLogging() {
    google::InitGoogleLogging("SFM");
}

std::once_flag initLoggingFlag;

struct SimpleReprojectionError {
    SimpleReprojectionError(double observed_x, double observed_y) :
            observed_x(observed_x), observed_y(observed_y) {
    }
    template<typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    const T* const focal,
                    T* residuals) const {
        T p[3];

        ceres::AngleAxisRotatePoint(camera, point, p);


        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];


        const T xp = p[0] / p[2];
        const T yp = p[1] / p[2];


        const T predicted_x = *focal * xp;
        const T predicted_y = *focal * yp;


        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SimpleReprojectionError, 2, 6, 3, 1>(
                new SimpleReprojectionError(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};

void BundleAdjustment::processBundleAdjustment(PointCloud &point_cloud, std::vector<cv::Matx34f> &camera_poses,
                                               cv::Mat &camera_parameters,
                                               const std::vector<Features> &images_features)
{
    std::call_once(initLoggingFlag, initLogging);


    ceres::Problem problem;

    typedef cv::Matx<double, 1, 6> CameraVector;
    std::vector<CameraVector> cameraPoses6d;
    cameraPoses6d.reserve(camera_poses.size());
    for (size_t i = 0; i < camera_poses.size(); i++) {
        const cv::Matx34f& pose = camera_poses[i];

        if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {

            cameraPoses6d.emplace_back();
            continue;
        }
        cv::Vec3f t(pose(0, 3), pose(1, 3), pose(2, 3));
        cv::Matx33f R = pose.get_minor<3, 3>(0, 0);
        float angleAxis[3];
        ceres::RotationMatrixToAngleAxis<float>(R.t().val, angleAxis);

        cameraPoses6d.emplace_back(
                angleAxis[0],
                angleAxis[1],
                angleAxis[2],
                t(0),
                t(1),
                t(2));
    }


    double focal = camera_parameters.at<float>(0, 0);

    std::vector<cv::Vec3d> points3d(point_cloud.size());

    for (int i = 0; i < point_cloud.size(); i++) {
        const auto& p = point_cloud[i];
        points3d[i] = cv::Vec3d(p.pt.x, p.pt.y, p.pt.z);

        for (const auto& kv : p.images) {

            cv::Point2f p2d = images_features[kv.first].points2D[kv.second];


            p2d.x -= camera_parameters.at<float>(0, 2);
            p2d.y -= camera_parameters.at<float>(1, 2);

            ceres::CostFunction* cost_function = SimpleReprojectionError::Create(p2d.x, p2d.y);

            problem.AddResidualBlock(cost_function,
                                     NULL /* squared loss */,
                                     cameraPoses6d[kv.first].val,
                                     points3d[i].val,
                                     &focal);
        }
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;
    options.eta = 1e-2;
    options.max_solver_time_in_seconds = 10;
    options.logging_type = ceres::LoggingType::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";



    camera_parameters.at<float>(0, 0) = focal;
    camera_parameters.at<float>(1, 1) = focal;

    for (size_t i = 0; i < camera_poses.size(); i++) {
        auto& pose = camera_poses[i];
        auto poseBefore = pose;

        if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {

            continue;
        }


        double rotationMat[9] = { 0 };
        ceres::AngleAxisToRotationMatrix(cameraPoses6d[i].val, rotationMat);

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                pose(c, r) = rotationMat[r * 3 + c];
            }
        }

        pose(0, 3) = cameraPoses6d[i](3);
        pose(1, 3) = cameraPoses6d[i](4);
        pose(2, 3) = cameraPoses6d[i](5);
    }

    for (int i = 0; i < point_cloud.size(); i++) {
        point_cloud[i].pt.x = points3d[i](0);
        point_cloud[i].pt.y = points3d[i](1);
        point_cloud[i].pt.z = points3d[i](2);
    }
}