"""
CS 6384 Homework 3 Programming
Epipolar Geometry
"""

from turtle import shape
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


#TODO
# use your backproject function in homework 1, problem 2
from backproject import backproject
    
    
# read rgb, depth, mask and meta data from files
def read_data(file_index):

    # read the image in data
    # rgb image
    rgb_filename = 'data/%06d-color.jpg' % file_index
    im = cv2.imread(rgb_filename)
    
    # depth image
    depth_filename = 'data/%06d-depth.png' % file_index
    depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    depth = depth / 1000.0
    
    # read the mask image
    mask_filename = 'data/%06d-label-binary.png' % file_index
    mask = cv2.imread(mask_filename)
    mask = mask[:, :, 0]
    
    # erode the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    
    # load matedata
    meta_filename = 'data/%06d-meta.mat' % file_index
    meta = scipy.io.loadmat(meta_filename)
    
    return im, depth, mask, meta
    
    
#TODO: implement this function to compute the fundamental matrix
# Follow lecture 11, the 8-point algorithm
# xy1 and xy2 are with shape (n, 2)
def compute_fundamental_matrix(xy1, xy2):
    pass
    # step 1: construct the A matrix
    n = len(xy1)
    a = []
    
    x1 = xy1.T[0]
    x2 = xy2.T[0]

    y1 = xy1.T[1]
    y2 = xy2.T[1]

    c1 = np.multiply(x2,x1)
    c2 = np.multiply(x2,y1)
    c3 = x2
    c4 = np.multiply(y2,x1)
    c5 = np.multiply(y2,y1)
    c6 = y2
    c7 = x1
    c8=y1
    c9=np.ones(n)
    
    a = np.vstack([c1,c2,c3,c4,c5,c6,c7,c8,c9]).T

    # print(a)
        
    
    
    # step 2: SVD of A
    # use numpy function for SVD
    u,d,vt = np.linalg.svd(a)
    
    # step 3: get the last column of V
    v = vt.T
    F = v[:,-1]
    
    F=np.reshape(F,(3,3))

    # return F
    # step 4: SVD of F
    uf,df,vft = np.linalg.svd(F)
    
    # step 5: mask the last element of singular value of F
    mask = [1,1,0]
    df[-1]=0

       
    # step 6: reconstruct F
    f = uf@np.diag(df)@vft 
    # f = np.reshape(f,9)

    return f


# main function
if __name__ == '__main__':

    # read image 1
    im1, depth1, mask1, meta1 = read_data(6)
    
    # read image 2
    im2, depth2, mask2, meta2 = read_data(7)
    
    # intrinsic matrix
    intrinsic_matrix = meta1['intrinsic_matrix']
    print('intrinsic_matrix')
    print(intrinsic_matrix)
        
    # get the point cloud from image 1
    pcloud = backproject(depth1, intrinsic_matrix)
    
    # find the boundary of the mask 1
    boundary = np.where(mask1 > 0)
    x1 = np.min(boundary[1])
    x2 = np.max(boundary[1])
    y1 = np.min(boundary[0])
    y2 = np.max(boundary[0])
    
    # sample n pixels (x, y) inside the bounding box of the cracker box
    # due to the randomness here, you may not get the same figure as mine
    # this is fine as long as your result is correct    
    n = 10
    height = im1.shape[0]
    width = im1.shape[1]
    x = np.random.randint(x1, x2, n)
    y = np.random.randint(y1, y2, n)
    index = np.zeros((n, 2), dtype=np.int32)
    index[:, 0] = x
    index[:, 1] = y
    print(index, index.shape)

    # get the coordinates of the n pixels
    pc1 = np.ones((4, n), dtype=np.float32)
    for i in range(n):
        x = index[i, 0]
        y = index[i, 1]
        print(x, y)
        pc1[:3, i] = pcloud[y, x, :]
    print('pc1', pc1)
    
    # filter zero depth pixels
    ind = pc1[2, :] > 0
    pc1 = pc1[:, ind]
    index = index[ind]
    xy1 = index
    # xy1 is a set of pixels on image 1
    # we will find the correspondences of these pixels

    
    
    #TODO
    # use your code from homework 1, problem 3 to find the correspondences of xy1
    # let the corresponding pixels on image 2 be xy2 with shape (n, 2)
    pcloud = backproject(depth1,intrinsic_matrix)

    pts =[]
    for x,y in index:
        # points3d[(x,y)]= pcloud[x,y,:]
        pts.append(pcloud[y,x,:])
    # print("====",points3d)
    # exit 
    # Step 2: transform the points to the camera of image 2 using the camera poses in the meta data
    RT1 = meta1['camera_pose']
    RT2 = meta2['camera_pose']
    rt1_inv = np.linalg.inv(RT1)
    rt2_inv = np.linalg.inv(RT2)
    # print(RT1.shape, RT2.shape)
    #     org = matmul(inv(rt1),p1) 
    #     org2=    matmul(org,rt2)        
    #     matmul(k,org2)

    
    tr1 = []
    for point in pts:
        tmp = np.append(point,1)
        tr = np.matmul(rt1_inv , tmp  )
        tr2 = np.matmul(RT2,tr)
        p = np.column_stack((np.eye(3),np.zeros(3)))
        p2d = np.matmul(np.matmul(intrinsic_matrix,p), tr2)
        x,y = p2d[0]//p2d[2],p2d[1]//p2d[2] 
        # print("******************")
        # print("shape p2d:",p2d)
        # print("x,y",x,y)
        # print("******************")
        tr1.append(np.array([int(x),int(y)]))

    # org = np.matmul()
    # print("******************")
    # print("******************")
    # print(tr1)
    # print("******************")
    # print("******************")
    # Step 3: project the transformed 3D points to the second image
    # support the output of this step is x2d with shape (2, n) which will be used in the following visuali zation
    x2d=np.array(tr1).T
    xy2 = x2d.T
    
    # transform the points to another camera
    RT1 = meta1['camera_pose']
    RT2 = meta2['camera_pose']
    print(RT1.shape, RT2.shape)
    
    #TODO
    # implement this function: compute fundamental matrix
    F = compute_fundamental_matrix(xy1, xy2)
    
    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image 1 and sampled pixels
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(im1[:, :, (2, 1, 0)])
    ax.set_title('image 1: correspondences', fontsize=15)
    plt.scatter(x=xy1[:, 0], y=xy1[:, 1], c='y', s=20)
    
    # show RGB image 2 and sampled pixels
    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(im2[:, :, (2, 1, 0)])
    ax.set_title('image 2: correspondences', fontsize=15)
    plt.scatter(x=xy2[:, 0], y=xy2[:, 1], c='g', s=20)
    
    # show three pixels on image 1
    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(im1[:, :, (2, 1, 0)])
    ax.set_title('image 1: sampled pixels', fontsize=15)
    
    # compute epipolar lines of three sampled points
    px = 233
    py = 145
    p = np.array([px, py, 1]).reshape((3, 1))
    l1 = np.matmul(F, p)
    print(p.shape)
    print(l1) 
    plt.scatter(x=px, y=py, c='r', s=40)
    
    px = 240
    py = 245
    p = np.array([px, py, 1]).reshape((3, 1))
    l2 = np.matmul(F, p)
    plt.scatter(x=px, y=py, c='g', s=40)
    
    px = 326
    py = 268
    p = np.array([px, py, 1]).reshape((3, 1))
    l3 = np.matmul(F, p)
    plt.scatter(x=px, y=py, c='b', s=40)    
    
    # draw the epipolar lines of the three pixels
    ax = fig.add_subplot(2, 2, 4)
    plt.imshow(im2[:, :, (2, 1, 0)])
    ax.set_title('image 2: epipolar lines', fontsize=15)
    
    for x in range(width):
        y1 = (-l1[0] * x - l1[2]) / l1[1]
        if y1 > 0 and y1 < height-1:
            plt.scatter(x, y1, c='r', s=1)
            
        y2 = (-l2[0] * x - l2[2]) / l2[1]
        if y2 > 0 and y2 < height-1:
            plt.scatter(x, y2, c='g', s=1)
            
        y3 = (-l3[0] * x - l3[2]) / l3[1]
        if y3 > 0 and y3 < height-1:
            plt.scatter(x, y3, c='b', s=1)                        
                  
    plt.show()
