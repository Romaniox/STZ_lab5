import cv2
from cv2 import aruco
import yaml
import numpy as np
from pathlib import Path

from enum import Enum


class Position(Enum):
    NOTHING = 0
    FRONT = 1
    BACK = 2
    LEFT = 3
    RIGHT = 4
    TOP = 5


class CubeFace:
    def __init__(self, pt1=0.0, pt2=0.0, pt3=0.0, pt4=0.0, color=(128, 128, 128), is_top=False):
        self.ld = pt1  # left-down coord 2d
        self.rd = pt2  # right-down coord 2d
        self.ru = pt3  # left-up coord 2d
        self.lu = pt4  # right-up coord 2d

        self.center = 0  # center coord 2d

        self.color = color

        self.contours = np.array([])

        self.is_shown = False
        self.is_top = is_top

        self.pos = Position.NOTHING

    def rewrite(self, pt1, pt2, pt3, pt4):
        self.pos = Position.NOTHING
        self.is_shown = False

        if self.is_top:
            self.pos = Position.TOP

        self.ld = pt1
        self.rd = pt2
        self.ru = pt3
        self.lu = pt4

        self.contours = np.array([self.ld, self.rd, self.ru, self.lu])

        moments = cv2.moments(self.contours)
        try:
            self.center = np.array([(moments['m10'] / moments['m00']), (moments['m01'] / moments['m00'])]).astype(int)
        except ZeroDivisionError:
            self.center = np.array([0, 0])

    def step(self, img, front_face, top_face):
        self.__check_shown(front_face, top_face)

        if self.is_shown or self.pos is Position.TOP:
            self.__draw_face(img)
            self.__draw_contour(img)

    def __check_shown(self, front_face, top_face):
        if self.pos is Position.FRONT:
            self.is_shown = True
            return

        if self.pos is Position.LEFT or self.pos is Position.RIGHT:
            center = tuple(map(int, self.center))
            if (cv2.pointPolygonTest(top_face.contours, center, measureDist=False) != 1.0) \
                    and (cv2.pointPolygonTest(
                front_face.contours, center, measureDist=False) != 1.0):
                self.is_shown = True

    # def check_shown(self):
    #     if (self.ld[0] < self.rd[0] and self.ld[1] < self.rd[1]) or (
    #             self.ld[0] < self.rd[0] and self.ld[1] > self.rd[1]):
    #         self.is_shown = True
    #     else:
    #         self.is_shown = False

    def __draw_contour(self, img):
        thickness = 7
        cv2.line(img, self.ld, self.rd, (0, 0, 0), thickness)
        cv2.line(img, self.rd, self.ru, (0, 0, 0), thickness)
        cv2.line(img, self.ru, self.lu, (0, 0, 0), thickness)
        cv2.line(img, self.lu, self.ld, (0, 0, 0), thickness)

    def __draw_face(self, img):
        face_ctr = np.array([self.ld, self.rd, self.ru, self.lu])
        cv2.fillPoly(img, [face_ctr], self.color)


def draw_vertexes(img, imgpts):
    for pt in imgpts:
        pt = pt.astype(int)
        img = cv2.circle(img, pt, 8, (0, 0, 255), -1)
    return img


def get_positions(faces):
    faces_down_dot = {
        faces[0]: ((faces[0].ld + faces[0].rd) / 2)[1],
        faces[1]: ((faces[1].ld + faces[1].rd) / 2)[1],
        faces[2]: ((faces[2].ld + faces[2].rd) / 2)[1],
        faces[3]: ((faces[3].ld + faces[3].rd) / 2)[1],
    }
    front_face = max(faces_down_dot, key=faces_down_dot.get)
    front_face.pos = Position.FRONT

    for face in faces:
        if face.pos is Position.FRONT:
            continue
        if np.all(face.rd == front_face.ld) and np.all(face.ru == front_face.lu):
            face.pos = Position.LEFT
        elif np.all(face.ld == front_face.rd) and np.all(face.lu == front_face.ru):
            face.pos = Position.RIGHT
        else:
            face.pos = Position.BACK


def draw_cube(img, imgpts, faces):
    faces[0].rewrite(imgpts[0], imgpts[1], imgpts[5], imgpts[4])
    faces[1].rewrite(imgpts[1], imgpts[2], imgpts[6], imgpts[5])
    faces[2].rewrite(imgpts[2], imgpts[3], imgpts[7], imgpts[6])
    faces[3].rewrite(imgpts[3], imgpts[0], imgpts[4], imgpts[7])

    faces[4].rewrite(imgpts[4], imgpts[5], imgpts[6], imgpts[7])

    get_positions(faces[:4])

    for face in faces:
        if face.pos == Position.TOP:
            top_face = face
        elif face.pos == Position.FRONT:
            front_face = face

    for face in faces:
        face.step(img, front_face, top_face)


if __name__ == '__main__':
    root = Path(__file__).parent.absolute()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

    arucoParams = aruco.DetectorParameters_create()

    # camera = cv2.VideoCapture(0)
    # address = 'http://192.168.0.65:8080/video'
    # camera.open(address)

    with open('calibration.yaml', 'r') as f:
        loadeddict = yaml.safe_load(f)

    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

    axis = np.float32([[0.35, -0.35, 0], [0.35, 0.35, 0], [-0.35, 0.35, 0], [-0.35, -0.35, 0],
                       [0.35, -0.35, 0.7], [0.35, 0.35, 0.7], [-0.35, 0.35, 0.7], [-0.35, -0.35, 0.7]])

    face_1 = CubeFace(color=(0, 0, 255))
    face_2 = CubeFace(color=(255, 0, 0))
    face_3 = CubeFace(color=(0, 255, 0))
    face_4 = CubeFace(color=(0, 255, 255))

    face_5 = CubeFace(color=(182, 0, 235), is_top=True)

    faces = [face_1, face_2, face_3, face_4, face_5]

    img = cv2.imread(r'D:\OpenCV\lab5\python\11.png')
    # ret, img = camera.read()
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    while True:
        # ret, img = camera.read()
        h, w = img.shape[:2]
        img = cv2.imread(r'D:\OpenCV\lab5\python\11.png')
        img_out = img.copy()
        img_out = cv2.undistort(img_out, mtx, dist, None, newcameramtx)
        img_aruco = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

        corners, _, _ = aruco.detectMarkers(img_aruco, aruco_dict, parameters=arucoParams)
        if len(corners):
            corners = np.array(corners)
            # img_aruco = aruco.drawDetectedMarkers(img_aruco, corners, ids, (0, 255, 0))
            for corner in corners:
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, 0.7, newcameramtx, dist)  # For a board
                # print("Rotation:", rvec)
                # print("Translation:", tvec)

                imgpts, _ = cv2.projectPoints(axis, rvec, tvec, newcameramtx, dist)
                imgpts = np.int32(imgpts).reshape(-1, 2)
                # img_out = draw_vertexes(img_out, imgpts)
                # img_out = cv2.drawFrameAxes(img_out, newcameramtx, dist, rvec, tvec, 1)

                draw_cube(img_out, imgpts, faces)

        if cv2.waitKey(2) == ord('q'):
            break

        cv2.imshow("Res", img_out)

    cv2.destroyAllWindows()
