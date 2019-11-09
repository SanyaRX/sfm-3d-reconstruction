//
// Created by Lenovo on 11/8/2019.
//

#ifndef SFM_3D_RECONSTRUCTION_BUNDLEADJUSTMENT_H
#define SFM_3D_RECONSTRUCTION_BUNDLEADJUSTMENT_H

#include "CommonUtilities.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>


class BundleAdjustment {

public:
    /**
     * Runs bundle adjustment algorithm
     * @param point_cloud - currently reconstructed points
     * @param camera_poses - camera poses in world coordinates
     * @param camera_parameters - camera intrinsic parameters
     * @param images_features - image features
     */
    static void processBundleAdjustment(PointCloud &point_cloud,
                                        std::vector<cv::Matx34f> &camera_poses,
                                        cv::Mat &camera_parameters,
                                        const std::vector<Features> &images_features);

};


#endif //SFM_3D_RECONSTRUCTION_BUNDLEADJUSTMENT_H
