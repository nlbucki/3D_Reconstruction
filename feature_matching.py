import numpy as np
import cv2
import silx.image.sift as sift
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 1

img1 = cv2.imread('images/im1/left.png',0)      # queryImage
img2 = cv2.imread('images/im1/right.png',0)      # trainImage

# img1 = img1[50:, 50:]
# img2 = img2[50:, 50:]

# Initiate SIFT detector
siftp = sift.SiftPlan(img1.shape, img1.dtype, devicetype="GPU")
kp1 = siftp.keypoints(img1)
siftp = sift.SiftPlan(img2.shape, img2.dtype, devicetype="GPU")
kp2 = siftp.keypoints(img2)
des1 = np.array([k[4] for k in kp1])
des2 = np.array([k[4] for k in kp2])
matchp = sift.MatchPlan()
matches = matchp.match(kp1,kp2)
src_pts, dst_pts = [], []

print np.shape(matches)
for m in matches:  
    src_pts.append([m[0][0], m[0][1]])
    dst_pts.append([m[1][0], m[1][1]])

src_pts = np.array(src_pts)
dst_pts = np.array(dst_pts)


plt.subplot(1,2,1)
plt.imshow(img1,'gray')
plt.scatter(src_pts[:,0], src_pts[:,1])
plt.subplot(1,2,2)
plt.imshow(img2,'gray')
plt.scatter(dst_pts[:,0], dst_pts[:,1])
plt.show()

# orb = cv2.ORB_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
# # print kp2[1]

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)

# good = []
# pts1 = []
# pts2 = []

# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         good.append(m)
#         pts2.append(kp2[m.trainIdx].pt)
#         pts1.append(kp1[m.queryIdx].pt)

# print good
# if len(good)>MIN_MATCH_COUNT:
   
# src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
# dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.FM_LMEDS)
matchesMask = mask.ravel().tolist()
print F

#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,F)

#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
# else:
#     print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#     matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,[],None,**draw_params)

plt.imshow(img3,'gray'),plt.show()