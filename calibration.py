import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8*6,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)*28.5

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./images/Camera_Calibration/*.JPG')
 
for fname in images:
    print fname
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    smallGray = cv2.resize(gray, (1500,1000))

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(smallGray, (8,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners = 4*corners
        cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.drawChessboardCorners(img, (8,6), corners, ret)
    # cv2.imshow('img', img)
    # cv2.resizeWindow('img', 1500, 1000)
    # cv2.waitKey()
 
cv2.destroyAllWindows()

error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
np.savez('calibration.npz', K=mtx, error=error)
print error
print mtx