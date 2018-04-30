import numpy as np
import cv2
import silx.image.sift as sift
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

def find_3d_points(P1, P2, matches):

    numMatches = matches.shape[0]
    points_3d = np.zeros((numMatches,3))
    errs = np.zeros((numMatches,1))
    for m in matches:
        A = np.zeros((4, 4)) 
        A[1,:] = np.array([P1[0,0]-m[0][0]*P1[2,0], P1[0,1]-m[0][0]*P1[2,1],
                          P1[0,2]-m[0][0]*P1[2,2], P1[0,3]-m[0][0]*P1[2,3]])
        # A(2,:) = [P1(2,1)-matches(i,2)*P1(3,1), P1(2,2)-matches(i,2)*P1(3,2), ...
        #           P1(2,3)-matches(i,2)*P1(3,3), P1(2,4)-matches(i,2)*P1(3,4)];
        # A(3,:) = [P2(1,1)-matches(i,3)*P2(3,1), P2(1,2)-matches(i,3)*P2(3,2), ...
        #           P2(1,3)-matches(i,3)*P2(3,3), P2(1,4)-matches(i,3)*P2(3,4)];
        # A(4,:) = [P2(2,1)-matches(i,4)*P2(3,1), P2(2,2)-matches(i,4)*P2(3,2), ...
        #           P2(2,3)-matches(i,4)*P2(3,3), P2(2,4)-matches(i,4)*P2(3,4)];
    #     [U,S,V] = svd(A);
    #     x = V(:,end)/V(4,end);
    #     points_3d(i,:) = x(1:3);
    #     rec_err_1 = sqrt((dot(P1(1,:),x)/dot(P1(3,:),x) - matches(i,1))^2 ...
    #                 + (dot(P1(2,:),x)/dot(P1(3,:),x) - matches(i,2))^2);
    #     rec_err_2 = sqrt((dot(P2(1,:),x)/dot(P2(3,:),x) - matches(i,3))^2 ...
    #                 + (dot(P2(2,:),x)/dot(P2(3,:),x) - matches(i,4))^2);
    #     errs(i) = 0.5*(rec_err_1 + rec_err_2);
    # end
    # err = mean(errs);
    return 0,0

# Load images and calibration matrix
img1 = cv2.imread('images/Reconstruction_Test/DSC_0466.JPG',0)      # queryImage
img2 = cv2.imread('images/Reconstruction_Test/DSC_0467.JPG',0)      # trainImage
img1 = cv2.resize(img1,(1500,1000))
img2 = cv2.resize(img2,(1500,1000))

K = np.load('calibration.npz')['K']

# Initiate SIFT detector and find matches in images
siftp = sift.SiftPlan(img1.shape, img1.dtype, devicetype="GPU")
kp1 = siftp.keypoints(img1)
siftp = sift.SiftPlan(img2.shape, img2.dtype, devicetype="GPU")
kp2 = siftp.keypoints(img2)
des1 = np.array([k[4] for k in kp1])
des2 = np.array([k[4] for k in kp2])
matchp = sift.MatchPlan()
matches = matchp.match(kp1,kp2)
src_pts, dst_pts = [], []

# Use matches to find fundamental matrix and essential matrix
for m in matches:  
    src_pts.append([m[0][0], m[0][1]])
    dst_pts.append([m[1][0], m[1][1]])

src_pts = np.array(src_pts)
dst_pts = np.array(dst_pts)

F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.FM_LMEDS)
matchesMask = mask.ravel().tolist()
E = np.matmul(K.T, np.matmul(F, K))

# Plot matches
# plt.subplot(1,2,1)
# plt.imshow(img1,'gray')
# plt.scatter(src_pts[:,0], src_pts[:,1])
# plt.subplot(1,2,2)
# plt.imshow(img2,'gray')
# plt.scatter(dst_pts[:,0], dst_pts[:,1])
# plt.show()

# Find rotation and translation of cameras
U, S, Vh = np.linalg.svd(E)
# print np.matmul(U, np.matmul(np.diag(S),Vh)), E
R90 = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2), 0],[np.sin(np.pi/2), np.cos(np.pi/2), 0], [0, 0, 1]])
t = np.array([U[:,2], -U[:,2]])
R = np.array([np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90.T))).T,
              -np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90.T))).T,
              np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90))).T,
              -np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90))).T])

# Find R2 and t2 from R,t such that largest number of points lie in front
# of the image planes of the two cameras
Rt1 = np.concatenate((np.eye(3),np.zeros((3,1))), axis=1)
P1 = np.matmul(K,Rt1)

# The number of points in front of the image planes for all combinations
num_points = np.zeros((t.shape[0],R.shape[0]))
# the reconstruction error for all combinations
errs = np.full((t.shape[0],R.shape[0]), np.inf)

for ti in t:
    for ri in R:
        P2 = np.matmul(K, np.concatenate((ri,ti.reshape((3,1))), axis=1))      
        points_3d, err = find_3d_points(P1,P2,matches)
        
#         Z1 = points_3d(:,3);
#         Z2 = R2(3,:)*points_3d'+t2(3);Z2 = Z2';
#         num_points(ti,ri) = sum(Z1>0 & Z2>0);
        
                
#     end
# end