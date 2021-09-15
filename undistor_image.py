import numpy as np
import cv2


def load_parameter(img):
    parameters = np.load("C.npz")
    mtx = parameters["mtx"]
    dist = parameters["dist"]
    rvecs = parameters["rvecs"]
    tvecs = parameters["tvecs"]
    print(mtx)
    img = cv2.imread(img, 0)
    h, w = img.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return dst


img = "2.png"

img = load_parameter(img)
cv2.imshow("undistortion", img)
cv2.waitKey(2222000)