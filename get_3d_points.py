import numpy as np
import cv2
import silx.image.sift as sift
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Get 3d points for given matches and matrices, return 3d points and 
# reconstruction error
def find_3d_points(P1, P2, matches):

    numMatches = matches.shape[0]
    points_3d = np.zeros((numMatches,3))
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
        rec_err_1 = np.sqrt((P1[0,:].dot(x)/P1[2,:].dot(x) - m[0][0])**2
                         + (P1[1,:].dot(x)/P1[2,:].dot(x) - m[0][1])**2)
        rec_err_2 = np.sqrt((P2[0,:].dot(x)/P2[2,:].dot(x) - m[1][0])**2
                         + (P2[1,:].dot(x)/P2[2,:].dot(x) - m[1][1])**2)
        errs[i] = 0.5*(rec_err_1 + rec_err_2)
    err = np.mean(errs)
    return points_3d, err

def get_3d_points(im1, im2, K, plotMatches=False):

    # Initiate SIFT detector get keypoints in images
    siftp = sift.SiftPlan(im1.shape, im1.dtype, devicetype="GPU")
    kp1 = siftp.keypoints(im1)
    siftp = sift.SiftPlan(im2.shape, im2.dtype, devicetype="GPU")
    kp2 = siftp.keypoints(im2)

    # Extract descriptors from keypoints
    des1 = np.array([k[4] for k in kp1])
    des2 = np.array([k[4] for k in kp2])

    # Find matching keypoints in images
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
        plt.subplot(1,2,1)
        plt.imshow(im1,'gray')
        plt.scatter(src_pts[:,0], src_pts[:,1])
        plt.subplot(1,2,2)
        plt.imshow(im2,'gray')
        plt.scatter(dst_pts[:,0], dst_pts[:,1])
        plt.show()

    # Find all possible rotations and translations of camera 2
    U, S, Vh = np.linalg.svd(E)
    R90 = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2), 0],[np.sin(np.pi/2), np.cos(np.pi/2), 0], [0, 0, 1]])
    t = np.array([U[:,2], -U[:,2]])
    R = np.array([np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90.T))).T,
                -np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90.T))).T,
                np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90))).T,
                -np.matmul(Vh.T, np.linalg.inv(np.matmul(U,R90))).T])

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
    points, err = find_3d_points(P1,P2,matches)
    return points, err, R[ri], t[ti]

if __name__ == '__main__':

    # Load images and calibration matrix
    img1 = cv2.imread('images/Reconstruction_Test/DSC_0466.JPG',0)
    img2 = cv2.imread('images/Reconstruction_Test/DSC_0467.JPG',0)
    img1 = cv2.resize(img1,(1500,1000))
    img2 = cv2.resize(img2,(1500,1000))

    cam_calib = np.load('calibration.npz')
    K = cam_calib['K']
    distCoeffs = cam_calib['distortion']

    h1,w1 = img1.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,distCoeffs,(w1,h1),1,(w1,h1))
    img1_undistort = cv2.undistort(img1, K, distCoeffs, None, newcameramtx)
    h2,w2 = img2.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K,distCoeffs,(w2,h2),1,(w2,h2))
    img2_undistort = cv2.undistort(img2, K, distCoeffs, None, newcameramtx)
    # print roi
    # plt.imshow(img1_undistort, 'gray')
    # plt.show()

    # crop the image
    # x,y,w,h = roi
    # img1_undistort = img1_undistort[y:y+h, x:x+w]
    plt.figure()
    plt.imshow(img1_undistort)
    plt.figure()
    plt.imshow(img2_undistort)
    plt.show()

    points,err,R,t = get_3d_points(img1_undistort, img2_undistort, K, plotMatches=True)
    np.savez('reconstruction_test', points=points, error=err, K=K, R=R, t=t)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], zs=points[:,2])
    plt.show()