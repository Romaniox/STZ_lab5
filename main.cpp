#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/imgproc.hpp>

#include "Cube.h"

static void readDetectorParameters(std::string filename, cv::Ptr<cv::aruco::DetectorParameters> &params) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        return;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
}

static bool readCalibration(const std::string &filename, cv::Mat &mtx, cv::Mat &dist) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;
    fs["camera_matrix"] >> mtx;
    fs["dist_coeff"] >> dist;
    return true;
}

void draw_vertex(cv::Mat &img, std::vector<cv::Point2d> &imgpts) {
    for (int i; i < imgpts.size(); i++) {
        cv::Point pt = imgpts[i];
        cv::circle(img, pt, 8, cv::Scalar(0, 0, 255), -1);
    }
}

void get_positions(std::vector<CubeFace *> &faces) {
    int max_y = 0;
    int num = 0;
    for (int i = 0; i < faces.size(); i++) {
        if (faces[i]->center.y > max_y) {
            max_y = faces[i]->center.y;
            num = i;
        }
    }
    CubeFace *front_face = faces[num];
    front_face->pos = FRONT;

    for (auto &face : faces) {
        if (face->pos == FRONT) {
            continue;
        }
        if (face->rd == front_face->ld && face->ru == front_face->lu) {
            face->pos = LEFT;
        } else if (face->ld == front_face->rd && face->lu == front_face->ru) {
            face->pos = RIGHT;
        } else {
            face->pos = BACK;
        }
    }
}

std::vector<cv::Point3d> create_cube_points(double marker_length) {
    double marker_length_half = marker_length / 2;

    std::vector<cv::Point3d> objectPoints(8);
    objectPoints[0] = cv::Point3d(marker_length_half, -marker_length_half, 0.0);
    objectPoints[1] = cv::Point3d(marker_length_half, marker_length_half, 0.0);
    objectPoints[2] = cv::Point3d(-marker_length_half, marker_length_half, 0.0);
    objectPoints[3] = cv::Point3d(-marker_length_half, -marker_length_half, 0.0);
    objectPoints[4] = cv::Point3d(marker_length_half, -marker_length_half, marker_length);
    objectPoints[5] = cv::Point3d(marker_length_half, marker_length_half, marker_length);
    objectPoints[6] = cv::Point3d(-marker_length_half, marker_length_half, marker_length);
    objectPoints[7] = cv::Point3d(-marker_length_half, -marker_length_half, marker_length);

    return objectPoints;
}

void draw_cube(cv::Mat &img, const std::vector<cv::Point2d> &imgpts, std::vector<CubeFace *> &faces) {
    faces[0]->rewrite(imgpts[0], imgpts[1], imgpts[5], imgpts[4]);
    faces[1]->rewrite(imgpts[1], imgpts[2], imgpts[6], imgpts[5]);
    faces[2]->rewrite(imgpts[2], imgpts[3], imgpts[7], imgpts[6]);
    faces[3]->rewrite(imgpts[3], imgpts[0], imgpts[4], imgpts[7]);

    faces[4]->rewrite(imgpts[4], imgpts[5], imgpts[6], imgpts[7]);

    std::vector<CubeFace *> tmp_vec;
    tmp_vec.push_back(faces[0]);
    tmp_vec.push_back(faces[1]);
    tmp_vec.push_back(faces[2]);
    tmp_vec.push_back(faces[3]);

    get_positions(tmp_vec);

    std::vector<cv::Point> top_face_contour;
    std::vector<cv::Point> front_face_contour;
    for (auto &face : faces) {
        if (face->pos == TOP) {
            top_face_contour = face->contour;
        } else if (face->pos == FRONT) {
            front_face_contour = face->contour;
        }
    }

    for (auto &face : faces) {
        face->step(img, front_face_contour, top_face_contour);
    }
}


int main() {
    cv::Ptr<cv::aruco::Dictionary> aruco_dict = cv::aruco::getPredefinedDictionary(
            cv::aruco::PREDEFINED_DICTIONARY_NAME(cv::aruco::DICT_6X6_250));


    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    std::vector<cv::Vec3d> rvec, tvec;

//    cv::Ptr<cv::aruco::DetectorParameters> arucoParams;
//    readDetectorParameters("detector_params.yaml", arucoParams);

    cv::Mat distanceCoefficients;

    double marker_length = 0.7;
    std::vector<cv::Point3d> objectPoints = create_cube_points(marker_length);
    std::vector<cv::Point2d> imagePoints;

    double mtx_array[3][3] = {{949.87525, 0.0,       617.77544},
                              {0.0,       946.73272, 475.53144},
                              {0.0,       0.0,       1.0}};
    cv::Mat mtx(3, 3, CV_64F, mtx_array);

    double dist_array[1][5] = {{0.02384, -0.17240, -0.00126, -0.00280, 0.18834}};
    cv::Mat dist(1, 5, CV_64F, dist_array);

    cv::Mat img_0;
    cv::VideoCapture camera(0);
    std::string address = "http://192.168.0.65:8080/video";
    camera.open(address);
    camera >> img_0;
    cv::Size size_img = cv::Size(img_0.size[1], img_0.size[0]);

    cv::Mat new_mtx(3, 3, CV_64F);
    new_mtx = cv::getOptimalNewCameraMatrix(mtx, dist, size_img, 1.0, size_img);

    cv::Point pt0 = cv::Point(0, 0);
    CubeFace face_1(pt0, pt0, pt0, pt0, cv::Scalar(0, 0, 255), false);
    CubeFace face_2(pt0, pt0, pt0, pt0, cv::Scalar(255, 0, 0), false);
    CubeFace face_3(pt0, pt0, pt0, pt0, cv::Scalar(0, 255, 0), false);
    CubeFace face_4(pt0, pt0, pt0, pt0, cv::Scalar(0, 255, 255), false);
    CubeFace face_5(pt0, pt0, pt0, pt0, cv::Scalar(182, 0, 255), true);

    std::vector<CubeFace *> faces;
    faces.push_back(&face_1);
    faces.push_back(&face_2);
    faces.push_back(&face_3);
    faces.push_back(&face_4);
    faces.push_back(&face_5);

    cv::Mat img_out, img_aruco;
    while (true) {
        camera >> img_0;

        if (img_0.empty()) {
            break;
        }

        cv::undistort(img_0, img_out, mtx, dist, new_mtx);
        cv::aruco::detectMarkers(img_out, aruco_dict, corners, ids);
        if (!ids.empty()) {
//            cv::aruco::drawDetectedMarkers(img_out, corners, ids, cv::Scalar(0, 255, 0));
            for (int i = 0; i < ids.size(); i++) {
                cv::aruco::estimatePoseSingleMarkers(corners, 0.7, new_mtx, dist, rvec, tvec);
                cv::projectPoints(objectPoints, rvec, tvec, new_mtx, dist, imagePoints);
//                cv::drawFrameAxes(img_out, mtx, dist, rvec, tvec, 1);
//                draw_vertex(img_out, imagePoints);
                draw_cube(img_out, imagePoints, faces);
            }
        }

        cv::imshow("Vid", img_out);
        char c = (char) cv::waitKey(2);
        if (c == 113)
            break;
    }
    return 0;
}
