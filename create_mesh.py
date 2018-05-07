import numpy as np
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import cv2

img1 = cv2.imread('images/Reconstruction_Test/DSC_0466.JPG',0)
img2 = cv2.imread('images/Reconstruction_Test/DSC_0467.JPG',0)
img1 = cv2.resize(img1,(1500,1000))
img2 = cv2.resize(img2,(1500,1000))
    
# points_3d = np.array([[0,0,0], [1.2,0,0], [0,2,0], [1.1,1,0.9], [0,0,1.5]])
calib_data = np.load('calibration.npz')
test_data = np.load('./images/Reconstruction_Test/reconstruction_test.npz')
print calib_data.files
print test_data.files

K = calib_data['K']
distCoeffs = calib_data['distortion']
R = test_data['R']
t = test_data['t']
points_3d = test_data['points']

# print distCoeffs
# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K, distCoeffs, K, distCoeffs, img1.shape[::-1], R, t, alpha=-1)
# map1, map2 = cv2.initUndistortRectifyMap(K, distCoeffs, R1, P1, img1.shape[::-1], cv2.CV_32FC1)
# img1Rect = cv2.remap(img1, map1, map2, cv2.INTER_LINEAR)

# plt.imshow(img1Rect,'gray')
# plt.figure()
# plt.imshow(roi2)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

tri = Delaunay(points_3d)
print points_3d[tri.simplices]
ax.plot(points_3d[:,0], points_3d[:,1], points_3d[:,2], '.r')

for tetra in points_3d[tri.simplices]:
    ax.plot(tetra[:,0],tetra[:,1],tetra[:,2],'b')
#     print tetra

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
