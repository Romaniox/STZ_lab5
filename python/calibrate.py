import cv2
from cv2 import aruco
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

root = Path(__file__).parent.absolute()
calib_imgs_path = root.joinpath('aruco_data')

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(7, 10, 1, 0.7, aruco_dict)
imboard = board.draw((1400, 2000))
arucoParams = aruco.DetectorParameters_create()

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
ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape,
                                                          None, None)

print("Camera matrix is \n", mtx,
      "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
with open("calibration1.yaml", "w") as f:
    yaml.dump(data, f)

cv2.destroyAllWindows()
