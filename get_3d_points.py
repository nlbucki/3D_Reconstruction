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
    siftp = sift.SiftPlan(img1.shape, img1.dtype, devicetype="CPU")
    kp1 = siftp.keypoints(img1)
    siftp = sift.SiftPlan(img2.shape, img2.dtype, devicetype="CPU")
    kp2 = siftp.keypoints(img2)

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
    E = np.matmul(K.T, np.matmul(F, K))

    # Plot matches if flag is set
    if plotMatches:
        plt.subplot(1,2,1)
        plt.imshow(img1,'gray')
        plt.scatter(src_pts[:,0], src_pts[:,1])
        plt.subplot(1,2,2)
        plt.imshow(img2,'gray')
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

    K = np.load('calibration.npz')['K']

    points,err,R,t = get_3d_points(img1, img2, K, plotMatches=False)
    np.savez('reconstruction_test', points=points, error=err, K=K, R=R, t=t)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], zs=points[:,2])
    plt.show()