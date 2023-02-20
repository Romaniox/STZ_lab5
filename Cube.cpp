#include "Cube.h"

CubeFace::CubeFace(cv::Point pt1, cv::Point pt2, cv::Point pt3, cv::Point pt4, cv::Scalar color, bool is_top) {
    this->ld = pt1;
    this->rd = pt2;
    this->ru = pt3;
    this->lu = pt4;

    this->color = std::move(color);
    this->center = cv::Point(0, 0);

    this->pos = NOTHING;
    this->is_shown = false;
    this->is_top = is_top;

    this->contour.reserve(4);
}


void CubeFace::rewrite(cv::Point pt1, cv::Point pt2, cv::Point pt3, cv::Point pt4) {
    this->pos = NOTHING;
    this->is_shown = false;

    if (this->is_top) {
        this->pos = TOP;
    }

    this->ld = pt1;
    this->rd = pt2;
    this->ru = pt3;
    this->lu = pt4;

    this->contour.clear();
    this->contour.push_back(this->ld);
    this->contour.push_back(this->rd);
    this->contour.push_back(this->ru);
    this->contour.push_back(this->lu);

    cv::Moments moments;
    moments = cv::moments(contour);
    this->center = cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);
}

void CubeFace::step(cv::Mat &img, const std::vector<cv::Point>& front_face_contour, const std::vector<cv::Point>& top_face_contour) {
    this->check_shown(front_face_contour, top_face_contour);

    if (this->is_shown || this->pos == TOP) {
        this->draw_face(img);
        this->draw_contour(img);
    }
}

void CubeFace::check_shown(const std::vector<cv::Point>& front_face_contour, const std::vector<cv::Point>& top_face_contour) {
    if (this->pos == FRONT) {
        this->is_shown = true;
        return;
    }

    if (this->pos == LEFT || this->pos == RIGHT) {
        if (cv::pointPolygonTest(top_face_contour, this->center, false) != 1.0 &&
            cv::pointPolygonTest(front_face_contour, this->center, false) != 1.0) {
            this->is_shown = true;
        }
    }
}

void CubeFace::draw_contour(cv::Mat &img) const {
    int thickness = 7;
    cv::Scalar color_line = cv::Scalar(0, 0, 0);
    cv::line(img, this->ld, this->rd, color_line, thickness);
    cv::line(img, this->rd, this->ru, color_line, thickness);
    cv::line(img, this->ru, this->lu, color_line, thickness);
    cv::line(img, this->lu, this->ld, color_line, thickness);
}

void CubeFace::draw_face(cv::Mat &img) const {
    cv::fillPoly(img, this->contour, this->color);
}
