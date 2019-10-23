//
// Created by Lenovo on 23.10.2019.
//

#include "CommonUtilities.h"

std::vector<cv::Mat> CommonUtilities::loadImages(const std::string &directory_path,
                                       const std::string &list_file_name,
                                       float resize_scale)
{
    std::ifstream fin;
    fin.open(directory_path + "\\" + list_file_name);
    std::string image_name;
    std::vector<cv::Mat> images;
    while (fin >> image_name)
    {
        cv::Mat image = cv::imread(directory_path + "\\" + image_name);
        if (resize_scale > 0 && resize_scale < 1)
            cv::resize(image, image, cv::Size(), resize_scale, resize_scale);
        images.push_back(image);
    }

    return images;
}

void CommonUtilities::drawImageMatches(const cv::Mat &first_image,
                                       const cv::Mat &second_image,
                                       const KeyPoints &first_image_key_points,
                                       const KeyPoints &second_image_key_points,
                                       const Matches &matches)
{
    cv::Mat img_matches;

    cv::drawMatches(first_image, first_image_key_points,
                second_image, second_image_key_points,
                matches, img_matches, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("Image Matches", img_matches);
    cv::waitKey();
}