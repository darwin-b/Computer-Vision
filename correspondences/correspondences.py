"""
CS 6384 Homework 1 Programming
Find correspondences of pixels using camera poses
"""

import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch import matmul

# first finish the backproject function in problem 2
from backproject import backproject

   
# read RGB image, depth image, mask image and meta data
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


# main function
if __name__ == '__main__':

    # read image 1
    im1, depth1, mask1, meta1 = read_data(6)
    
    # read image 2
    im2, depth2, mask2, meta2 = read_data(8)
    
    # intrinsic matrix. It is the same for both images
    intrinsic_matrix = meta1['intrinsic_matrix']
    print('intrinsic_matrix')
    print(intrinsic_matrix)
        
    # backproject the points for image 1
    pcloud = backproject(depth1, intrinsic_matrix)
    
    # sample 3 pixels in (x, y) format for image 1
    index = np.array([[257, 142], [363, 165], [286, 276]], dtype=np.int32)
    print(index, index.shape)
    
    # TODO finish the following steps to find the correspondences of the 3 pixels on image 2
    
    # Step 1: get the coordinates of 3D points for the 3 pixels from image 1
    # pixel1 = np.stack(index,np.array([1,1,1]))
    # points3d ={}
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
    print("******************")
    print("******************")
    print(tr1)
    print("******************")
    print("******************")
    # Step 3: project the transformed 3D points to the second image
    # support the output of this step is x2d with shape (2, n) which will be used in the following visuali zation
    x2d=np.array(tr1).T
    


    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image 1 and the 3 pixels
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(im1[:, :, (2, 1, 0)])
    ax.set_title('RGB image 1')
    plt.scatter(x=index[0, 0], y=index[0, 1], c='r', s=40)
    plt.scatter(x=index[1, 0], y=index[1, 1], c='g', s=40)
    plt.scatter(x=index[2, 0], y=index[2, 1], c='b', s=40)
    
    # show RGB image 2 and the corresponding 3 pixels
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(im2[:, :, (2, 1, 0)])
    ax.set_title('RGB image 2')
    plt.scatter(x=x2d[0, 0], y=x2d[1, 0].flatten(), c='r', s=40)
    plt.scatter(x=x2d[0, 1], y=x2d[1, 1].flatten(), c='g', s=40)
    plt.scatter(x=x2d[0, 2], y=x2d[1, 2].flatten(), c='b', s=40)
                  
    plt.show()
