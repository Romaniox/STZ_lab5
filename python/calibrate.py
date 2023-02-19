import cv2
from cv2 import aruco
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

root = Path(__file__).parent.absolute()
calibrate_camera = True

calib_imgs_path = root.joinpath('aruco_data')

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(7, 10, 1, 0.7, aruco_dict)
imboard = board.draw((1400, 2000))

arucoParams = aruco.DetectorParameters_create()

if calibrate_camera:
    img_list = []
    calib_fnms = calib_imgs_path.glob('*.jpg')
    print('Using ...', end='')
    for idx, fn in enumerate(calib_fnms):
        print(idx, '', end='')
        img = cv2.imread(str(root.joinpath(fn)))
        img_list.append(img)
        h, w, c = img.shape
    print('Calibration images')

    counter, corners_list, id_list = [], [], []
    first = True
    for im in tqdm(img_list):
        img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        if first:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list, ids))
        counter.append(len(ids))
    print('Found {} unique markers'.format(np.unique(ids)))

    counter = np.array(counter)
    print("Calibrating camera .... Please wait...")
    # mat = np.zeros((3,3), float)
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape,
                                                              None, None)

    print("Camera matrix is \n", mtx,
          "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration1.yaml", "w") as f:
        yaml.dump(data, f)

else:
    camera = cv2.VideoCapture(0)
    address = 'http://192.168.0.65:8080/video'
    camera.open(address)

    with open('calibration.yaml', 'r') as f:
        loadeddict = yaml.safe_load(f)

    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

    ret, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    pose_r, pose_t = [], []
    while True:
        ret, img = camera.read()
        img_aruco = img
        im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = im_gray.shape[:2]
        dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
        # cv2.imshow("original", img_gray)
        if len(corners) == 0:
            print("pass")

        else:
            # ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist)  # For a board
            # print("Rotation ", rvec, "Translation", tvec)
            if ret != 0:
                img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
                # img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec, tvec, 10)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cv2.imshow("World co-ordinate frame axes", img_aruco)
        cv2.waitKey(20)

cv2.destroyAllWindows()

pass
