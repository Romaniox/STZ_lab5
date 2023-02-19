#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/videoio.hpp>

enum Position {
    NOTHING,
    FRONT,
    BACK,
    LEFT,
    RIGHT,
    TOP
};

class CubeFace {
public:
    cv::Point ld;   // left-down coord 2d
    cv::Point rd;   // right-down coord 2d
    cv::Point ru;   // left-up coord 2d
    cv::Point lu;   // right-up coord 2d

    cv::Scalar color;
    cv::Point center;
    Position pos;
    std::vector<cv::Point> contour;
private:

    bool is_shown;
    bool is_top;
public:
    void rewrite(cv::Point pt1, cv::Point pt2, cv::Point pt3, cv::Point pt4);
    void step(cv::Mat &img, const std::vector<cv::Point>& front_face_contour, const std::vector<cv::Point>& top_face_contour);
private:
    void check_shown(const std::vector<cv::Point>& front_face_contour, const std::vector<cv::Point>& top_face_contour);
    void draw_contour(cv::Mat &img) const;
    void draw_face(cv::Mat &img) const;
public:
    CubeFace(cv::Point pt1, cv::Point pt2, cv::Point pt3, cv::Point pt4, cv::Scalar color, bool is_top);
    ~CubeFace() = default;
};
