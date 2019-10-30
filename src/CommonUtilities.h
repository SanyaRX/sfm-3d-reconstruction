//
// Created by Lenovo on 23.10.2019.
//

#ifndef SFM_3D_RECONSTRUCTION_COMMONUTILITIES_H
#define SFM_3D_RECONSTRUCTION_COMMONUTILITIES_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "vector"
#include <fstream>

typedef std::vector<cv::KeyPoint> KeyPoints;
typedef std::vector<cv::Mat> Descriptors;
typedef std::vector<cv::DMatch> Matches;
typedef std::vector<cv::Point2f> Points2D;
typedef std::vector<cv::Point3f> Points3D;
typedef struct FEATURES {
    KeyPoints key_points;
    Points2D points2D;
    cv::Mat descriptor;
} Features;

typedef struct IMAGE_MATCH {
    Matches matches;
    Points2D l_img_points;
    Points2D  r_img_points;
} ImageMatch;

typedef struct POINT3D{
    cv::Point3f pt;
    std::vector<unsigned int> images; // images that show the point
} Point3D;
typedef std::vector<Point3D> PointCloud;

typedef struct POINT_PROJECTION {
    cv::Point3f pt;
    int proj_idx;
} PointProjection;

class CommonUtilities {

public:
    /**
     * Reads images from directory according to file containing list of image names
     * @param directory_path - path to directory with images
     * @param list_file_name - name of file with image names list
     * @param resize_scale - float in interval (0, 1) defines resized image sizes. If out of (0, 1) then image won't be resized.
     * @return std::vector of images (as cv::Mat)
     */
    static std::vector<cv::Mat> loadImages(const std::string &directory_path,
                                           const std::string &list_file_name,
                                           float resize_scale = 0);

    /**
     * Draws image matches
     * @param first_image - first image
     * @param second_image - second image
     * @param first_image_key_points - first image key points
     * @param second_image_key_points - second image key points
     * @param matches - key points matches
     */
    static void drawImageMatches(const cv::Mat &first_image,
                          const cv::Mat &second_image,
                          const KeyPoints &first_image_key_points,
                          const KeyPoints &second_image_key_points,
                          const Matches &matches);


};


#endif //SFM_3D_RECONSTRUCTION_COMMONUTILITIES_H
