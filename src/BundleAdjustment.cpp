//
// Created by Lenovo on 11/8/2019.
//

#include "BundleAdjustment.h"

void initLogging() {
    google::InitGoogleLogging("SFM");
}

std::once_flag initLoggingFlag;

struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
            : observed_x(observed_x), observed_y(observed_y) {}
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];

        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = 1.0 + r2  * (l1 + l2  * r2);

        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        return true;
    }

    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};

void BundleAdjustment::processBundleAdjustment(PointCloud &point_cloud, std::vector<cv::Matx34f> &camera_poses,
                                               cv::Mat &camera_parameters,
                                               const std::vector<Features> &images_features)
{
    std::call_once(initLoggingFlag, initLogging);

    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;

    //Convert camera pose parameters from [R|t] (3x4) to [Angle-Axis (3), Translation (3), focal (1)] (1x7)
    typedef cv::Matx<double, 1, 6> CameraVector;
    std::vector<CameraVector> cameraPoses6d;
    cameraPoses6d.reserve(camera_poses.size());
    for (size_t i = 0; i < camera_poses.size(); i++) {
        const cv::Matx34f& pose = camera_poses[i];

        if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {
            //This camera pose is empty, it should not be used in the optimization
            cameraPoses6d.push_back(CameraVector());
            continue;
        }
        cv::Vec3f t(pose(0, 3), pose(1, 3), pose(2, 3));
        cv::Matx33f R = pose.get_minor<3, 3>(0, 0);
        float angleAxis[3];
        ceres::RotationMatrixToAngleAxis<float>(R.t().val, angleAxis); //Ceres assumes col-major...

        cameraPoses6d.push_back(CameraVector(
                angleAxis[0],
                angleAxis[1],
                angleAxis[2],
                t(0),
                t(1),
                t(2)));
    }

    //focal-length factor for optimization
    double focal = camera_parameters.at<float>(0, 0);

    std::vector<cv::Vec3d> points3d(point_cloud.size());

    for (int i = 0; i < point_cloud.size(); i++) {
        const auto& p = point_cloud[i];
        points3d[i] = cv::Vec3d(p.pt.x, p.pt.y, p.pt.z);

        for (const auto& kv : p.images) {
            //kv.first  = camera index
            //kv.second = 2d feature index
            cv::Point2f p2d = images_features[kv.first].points2D[kv.second];

            //subtract center of projection, since the optimizer doesn't know what it is
            p2d.x -= camera_parameters.at<float>(0, 2);
            p2d.y -= camera_parameters.at<float>(1, 2);

            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(p2d.x, p2d.y);

            problem.AddResidualBlock(cost_function,
                                     NULL /* squared loss */,
                                     cameraPoses6d[kv.first].val,
                                     points3d[i].val,
                                     &focal);
        }
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
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



    //update optimized focal
    camera_parameters.at<float>(0, 0) = focal;
    camera_parameters.at<float>(1, 1) = focal;

    //Implement the optimized camera poses and 3D points back into the reconstruction
    for (size_t i = 0; i < camera_poses.size(); i++) {
        auto& pose = camera_poses[i];
        auto poseBefore = pose;

        if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {
            //This camera pose is empty, it was not used in the optimization
            continue;
        }

        //Convert optimized Angle-Axis back to rotation matrix
        double rotationMat[9] = { 0 };
        ceres::AngleAxisToRotationMatrix(cameraPoses6d[i].val, rotationMat);

        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                pose(c, r) = rotationMat[r * 3 + c]; //`rotationMat` is col-major...
            }
        }

        //Translation
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