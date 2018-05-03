import numpy as np
import cv2
import silx.image.sift as sift
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Get 3d points for given matches and matrices, return 3d points and 
# reconstruction error
def find_3d_points(P1, P2, matches, plotReprojection=False):

    numMatches = matches.shape[0]
    points_3d = np.zeros((numMatches,3))
    proj_points1 = np.zeros((numMatches,2))
    proj_points2 = np.zeros((numMatches,2))
    errs = np.zeros((numMatches,1))
    for i,m in enumerate(matches):
        A = np.zeros((4, 4)) 
        A[0,:] = np.array([P1[0,0]-m[0][0]*P1[2,0], P1[0,1]-m[0][0]*P1[2,1],
                          P1[0,2]-m[0][0]*P1[2,2], P1[0,3]-m[0][0]*P1[2,3]])
        A[1,:] = np.array([P1[1,0]-m[0][1]*P1[2,0], P1[1,1]-m[0][1]*P1[2,1],
                           P1[1,2]-m[0][1]*P1[2,2], P1[1,3]-m[0][1]*P1[2,3]])
        A[2,:] = np.array([P2[0,0]-m[1][0]*P2[2,0], P2[0,1]-m[1][0]*P2[2,1],
                           P2[0,2]-m[1][0]*P2[2,2], P2[0,3]-m[1][0]*P2[2,3]])
        A[3,:] = np.array([P2[1,0]-m[1][1]*P2[2,0], P2[1,1]-m[1][1]*P2[2,1],
                           P2[1,2]-m[1][1]*P2[2,2], P2[1,3]-m[1][1]*P2[2,3]])
        U,S,Vh = np.linalg.svd(A)
        x = Vh.T[:,3]/Vh.T[3,3]
        points_3d[i,:] = x[0:3]
        proj_points1[i,:] = P1[0:2,:].dot(x)/P1[2,:].dot(x)
        proj_points2[i,:] = P2[0:2,:].dot(x)/P2[2,:].dot(x)
        rec_err_1 = np.sqrt((P1[0,:].dot(x)/P1[2,:].dot(x) - m[0][0])**2
                         + (P1[1,:].dot(x)/P1[2,:].dot(x) - m[0][1])**2)
        rec_err_2 = np.sqrt((P2[0,:].dot(x)/P2[2,:].dot(x) - m[1][0])**2
                         + (P2[1,:].dot(x)/P2[2,:].dot(x) - m[1][1])**2)
        errs[i] = 0.5*(rec_err_1 + rec_err_2)
    err = np.mean(errs)
    
    if plotReprojection:
        ax1.scatter(proj_points1[:,0], proj_points1[:,1], s=0.5, color='red')
        ax2.scatter(proj_points2[:,0], proj_points2[:,1], s=0.5, color='red')
    return points_3d, err

def get_3d_points(im1, im2, K, plotMatches=False):

    # Initiate SIFT detector get keypoints in images
#     siftp = sift.SiftPlan(img1.shape, img1.dtype, devicetype="CPU")
#     kp1 = siftp.keypoints(img1)
#     siftp = sift.SiftPlan(img2.shape, img2.dtype, devicetype="CPU")
#     kp2 = siftp.keypoints(img2)
    
    orb = cv2.ORB_create()
    kp1 = orb.detect(im1, None)
    kp2 = orb.detect(im2, None)
    kp1, des1 = orb.compute(im1, kp1)
    kp2, des2 = orb.compute(im2, kp2)
    
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    # Extract descriptors from keypoints
#     des1 = np.array([k[4] for k in kp1])
#     des2 = np.array([k[4] for k in kp2])

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     bf_matches = bf.match(des1,des2)
#     bf_matches = sorted(bf_matches, key = lambda x:x.distance)
#     bf_matches = bf_matches[:10]
    draw_matches(im2,kp2,im1,kp1,good)
    
    matches = []
    for m in good:
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx
        p1 = kp1[img1_idx].pt
        p2 = kp2[img2_idx].pt
        matches.append((p1,p2))
    matches = np.array(matches)

    # Find matching keypoints in images
#     matchp = sift.MatchPlan()
#     matches = matchp.match(kp1,kp2)
    src_pts, dst_pts = [], []

    # Use matches to find fundamental matrix and essential matrix
    for m in matches:  
        src_pts.append([m[0][0], m[0][1]])
        dst_pts.append([m[1][0], m[1][1]])

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)

    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
    
    # Calculate residual errors
    n = src_pts.shape[0]
    x1_h = np.vstack((src_pts.T, np.ones((1,n))))
    x2_h = np.vstack((dst_pts.T, np.ones((1,n))))
    res_err = 0;
    for i in range(n):
        d12 = abs(x2_h[:,[i]].T.dot(F.dot(x1_h[:,[i]])))/np.linalg.norm(F.dot(x1_h[:,[i]]), 2)
        d21 = abs(x1_h[:,[i]].T.dot(F.dot(x2_h[:,[i]])))/np.linalg.norm(F.dot(x2_h[:,[i]]), 2)
        res_err += d12**2 + d21**2
    print 'Residual error: ', res_err
    
#     matchesMask = mask.ravel().tolist()
    # retval, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts, dst_pts, F, (1000,1500))
    # im1 = cv2.warpPerspective(im1, H1, (1500,1000))
    # im2 = cv2.warpPerspective(im2, H2, (1500,1000))
    # stereo = cv2.StereoBM_create(numDisparities=32, blockSize=11)
    # disparity = stereo.compute(im1,im2)
    # plt.imshow(disparity, 'gray')
    # plt.figure()

    # plt.imshow(cv2.warpPerspective(im1, H1, (1500,1000)), 'gray')
    # plt.figure()
    # plt.imshow(cv2.warpPerspective(im2, H2, (1500,1000)), 'gray')
    # plt.show()
    E = np.matmul(K.T, np.matmul(F, K))

    # Plot matches if flag is set
    if plotMatches:
        ax1.imshow(im1,'gray')
        ax1.scatter(src_pts[:,0], src_pts[:,1], s=0.5)
        ax2.imshow(im2,'gray')
        ax2.scatter(dst_pts[:,0], dst_pts[:,1], s=0.5)

    # Find all possible rotations and translations of camera 2
    U, S, Vh = np.linalg.svd(E)
    R90 = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2), 0],[np.sin(np.pi/2), np.cos(np.pi/2), 0], [0, 0, 1]])
    t = np.array([U[:,2], -U[:,2]])
    R = np.array([np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90.T))).T,
                -np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90.T))).T,
                np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90))).T,
                -np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90))).T])
#     R = np.array([np.matmul(U, np.matmul(R90, Vh.T)),
#                 -np.matmul(U, np.matmul(R90, Vh.T)),
#                 np.matmul(U, np.matmul(R90.T, Vh.T)),
#                 -np.matmul(U, np.matmul(R90.T, Vh.T))])

    # Construct P1 for camera 1
    Rt1 = np.concatenate((np.eye(3),np.zeros((3,1))), axis=1)
    P1 = np.matmul(K,Rt1)

    # The number of points in front of the image planes for all combinations
    num_points = np.zeros((t.shape[0],R.shape[0]))
    # The reconstruction error for all combinations (for debugging)
    # errs = np.full((t.shape[0],R.shape[0]), np.inf)

    # Find R2 and t2 from R,t such that largest number of points lie in front
    # of the image planes of the two cameras
    for i,ti in enumerate(t):
        for j,ri in enumerate(R):
            P2 = np.matmul(K, np.concatenate((ri,ti.reshape((3,1))), axis=1))      
            points_3d, err = find_3d_points(P1,P2,matches)
            
            Z1 = points_3d[:,2]
            Z2 = ri[2,:].dot(points_3d.T)+ti[2]
            Z2 = Z2.T
            num_points[i,j] = np.sum(np.logical_and(Z1>0, Z2>0))
            # errs[i,j] = err

    ti,ri = np.unravel_index(np.argmax(num_points), num_points.shape)
    P2 = np.matmul(K, np.concatenate((R[ri],t[ti].reshape((3,1))), axis=1))

    # Compute the 3D points with the final P2
    points, err = find_3d_points(P1,P2,matches, plotReprojection=True)
    return points, err, R[ri], t[ti]

def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to 
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    
    plt.figure(figsize=(15,15))
    plt.imshow(new_img,'gray')

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # Load images and calibration matrix
#     img1 = cv2.imread('images/Mouthwash/DSC_0590.JPG',0)
#     img2 = cv2.imread('images/Mouthwash/DSC_0591.JPG',0)
    img1 = cv2.imread('images/library1.jpg',0)
    img2 = cv2.imread('images/library1.jpg',0)
    img1 = cv2.resize(img1,(1500,1000))
    img2 = cv2.resize(img2,(1500,1000))

    cam_calib = np.load('calibration.npz')
    K = cam_calib['K']
    distCoeffs = cam_calib['distortion']

    h1,w1 = img1.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,distCoeffs,(w1,h1),1,(w1,h1))
    img1_undistort = cv2.undistort(img1, K, distCoeffs, None, newcameramtx)
    img2_undistort = cv2.undistort(img2, K, distCoeffs, None, newcameramtx)
    # print roi
    # plt.imshow(img1_undistort, 'gray')
    # plt.show()

    # crop the image
    # x,y,w,h = roi
    # img1_undistort = img1_undistort[y:y+h, x:x+w]
#     plt.figure()
#     plt.imshow(img1_undistort)
#     plt.figure()
#     plt.imshow(img2_undistort)

#     points, err, R, t = get_3d_points(img1_undistort, img2_undistort, newcameramtx, plotMatches=True)

    K = np.array([[-580., 0., 258.], [0., -539., 204.], [0., 0., 1.]])

    points, err, R, t = get_3d_points(img1, img2, K, plotMatches=True)
    print 'Reconstruction error: ', err
    np.savez('reconstruction_test', points=points, error=err, K=K, R=R, t=t)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], zs=points[:,2], s=0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(-2.5, 2.5)
    ax.set_ylim3d(-2.5,2.5)
    ax.set_zlim3d(0,5)
    plt.show()
    