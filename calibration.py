import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
gridx = 9
gridy = 6
objp = np.zeros((gridx*gridy,3), np.float32)
objp[:,:2] = np.mgrid[0:gridx,0:gridy].T.reshape(-1,2)*23.4

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./images/Camera_Calibration4/*.JPG')
 
for fname in images:
    print fname
    img = cv2.imread(fname)
    img = cv2.resize(img, (1500,1000))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (gridx,gridy), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11,11),(-1,-1),criteria)
        imgpoints.append(corners)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.drawChessboardCorners(img, (gridx,gridy), corners, ret)
    cv2.imshow('img', img)
    cv2.waitKey()
 
cv2.destroyAllWindows()

error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
np.savez('calibration.npz', error=error, K=mtx, distortion=dist)
print error
print mtx
print dist