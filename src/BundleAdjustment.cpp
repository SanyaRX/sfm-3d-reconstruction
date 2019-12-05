#include "BundleAdjustment.h"
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "CommonUtilities.h"
#include <mutex>


using namespace std;



namespace BundleAdjustUtils {

        void initLogging() {
            google::InitGoogleLogging("SFM");
        }

        std::once_flag initLoggingFlag;

    }
    using namespace BundleAdjustUtils;

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

        static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
            return (new ceres::AutoDiffCostFunction<SimpleReprojectionError, 2, 6, 3, 1>(
                    new SimpleReprojectionError(observed_x, observed_y)));
        }
        double observed_x;
        double observed_y;
    };

    void BundleAdjustment::processBundleAdjustment(
            PointCloud&                  point_cloud,
            std::vector<cv::Matx34d>&           camera_poses,
            CameraParameters &camera_parameters,
            const std::vector<Features>& images_features) {

        std::call_once(initLoggingFlag, initLogging);

        ceres::Problem problem;

        typedef cv::Matx<double, 1, 6> camera_vector;
        vector<camera_vector> camera_poses6d;
        camera_poses6d.reserve(camera_poses.size());

        for (size_t i = 0; i < camera_poses.size(); i++) {
            const cv::Matx34d &pose = camera_poses[i];

            if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {
                camera_poses6d.push_back(camera_vector());
                continue;
            }
            cv::Vec3d t(pose(0, 3), pose(1, 3), pose(2, 3));
            cv::Matx33d R = pose.get_minor<3, 3>(0, 0);
            double angle_axis[3];
            ceres::RotationMatrixToAngleAxis<double>(R.t().val, angle_axis);

            camera_poses6d.push_back(camera_vector(
                    angle_axis[0],
                    angle_axis[1],
                    angle_axis[2],
                    t(0),
                    t(1),
                    t(2)));
        }

        double focal = camera_parameters.k_matrix.at<double>(0, 0);
        vector<cv::Vec3d> points3d(point_cloud.size());

        for (int i = 0; i < point_cloud.size(); i++) {
            const Point3D &p = point_cloud[i];
            points3d[i] = cv::Vec3d(p.pt.x, p.pt.y, p.pt.z);

            for (const auto &kv : p.images) {
                cv::Point2d p2d = images_features[kv.first].points2D[kv.second];

                p2d.x -= camera_parameters.k_matrix.at<double>(0, 2);
                p2d.y -= camera_parameters.k_matrix.at<double>(1, 2);

                ceres::CostFunction *cost_function = SimpleReprojectionError::Create(p2d.x, p2d.y);

                problem.AddResidualBlock(cost_function,
                                         NULL /* squared loss */,
                                         camera_poses6d[kv.first].val,
                                         points3d[i].val,
                                         &focal);
            }
        }


        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 500;
        options.eta = 1e-3;
        options.max_solver_time_in_seconds = 30 ;
        options.logging_type = ceres::LoggingType::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << "\n";

        if (summary.termination_type != ceres::CONVERGENCE) {
            cerr << "Bundle adjustment failed." << endl;
            return;
        }


        camera_parameters.k_matrix.at<double>(0, 0) = focal;
        camera_parameters.k_matrix.at<double>(1, 1) = focal;

        for (size_t i = 0; i < camera_poses.size(); i++) {
            cv::Matx34d &pose = camera_poses[i];
            cv::Matx34d poseBefore = pose;

            if (pose(0, 0) == 0 and pose(1, 1) == 0 and pose(2, 2) == 0) {

                continue;
            }


            double rotation_mat[9] = {0};
            ceres::AngleAxisToRotationMatrix(camera_poses6d[i].val, rotation_mat);

            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    pose(c, r) = rotation_mat[r * 3 + c];
                }
            }


            pose(0, 3) = camera_poses6d[i](3);
            pose(1, 3) = camera_poses6d[i](4);
            pose(2, 3) = camera_poses6d[i](5);
        }

        for (int i = 0; i < point_cloud.size(); i++) {
            point_cloud[i].pt.x = points3d[i](0);
            point_cloud[i].pt.y = points3d[i](1);
            point_cloud[i].pt.z = points3d[i](2);
        }


}