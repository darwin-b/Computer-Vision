"""
CS 6384 Homework 3 Programming
Triangulation
"""

import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


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


    
#TODO: implement this function for triangulation
# Follow lecture 11, slide 28: solve the optimziation problem to find the 3D point X
# RT1, RT2 are camera poses with shape (4, 4)
# K is the camera intrinsic matrix with shape (3, 3)
# xy1 and xy2 are with shape (n, 2)
def triangulation(RT1, RT2, K, xy1, xy2):

    print(xy1.shape, xy2.shape)
    n = xy1.shape[0]
    points = np.zeros((n, 3), dtype=np.float32)
    
    # do the triangulation for each correspondence
    for i in range(n):
    
        # HINT: use the least_squares function from scipy.optimize to solve the optimization problem
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        
        
    return points
    
    
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    


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
    
    # find the foreground of the mask
    foreground = np.where(mask1 > 0)

    # sample n pixels (x, y) on the cracker box
    # due to the randomness here, you may not get the same figure as mine
    # this is fine as long as your result is correct
    n = 30
    l = len(foreground[0])
    index = np.random.permutation(l)[:n]
    x = foreground[1][index]
    y = foreground[0][index]
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
    
    # filter zero depth pixele
    ind = pc1[2, :] > 0
    pc1 = pc1[:, ind]
    index = index[ind]
    xy1 = index
    # xy1 is a set of pixels on image 1
    # we will find the correspondences of these pixels    
    
    # transform the points to another camera
    RT1 = meta1['camera_pose']
    RT2 = meta2['camera_pose']
    print(RT1.shape, RT2.shape)
    
    #TODO
    # use your code from homework 1, problem 3 to find the correspondences of xy1
    # let the corresponding pixels on image 2 be xy2 with shape (n, 2)
    
    #TODO: implement this function for triangulation
    points = triangulation(RT1, RT2, intrinsic_matrix, xy1, xy2)
    
    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image 1 and pixels
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(im1[:, :, (2, 1, 0)])
    ax.set_title('image 1: correspondences', fontsize=15)
    plt.scatter(x=xy1[:, 0], y=xy1[:, 1], c='y', s=20)
    
    # show RGB image 2 and pixels
    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(im2[:, :, (2, 1, 0)])
    ax.set_title('image 2: correspondences', fontsize=15)
    plt.scatter(x=xy2[:, 0], y=xy2[:, 1], c='g', s=20)
    
    # show 3D points from triangulation
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='.', color='y', s=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D points from triangulation', fontsize=15)   
    set_axes_equal(ax) 
    
    # show 3D points from depth and camera pose
    # the two sets of 3D points should be close to verify your triangulation
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    RT_delta = np.linalg.inv(RT1)
    pc = np.matmul(RT_delta, pc1)
    print(pc.shape)
    pc[0, :] /= pc[3, :]
    pc[1, :] /= pc[3, :]
    pc[2, :] /= pc[3, :]
    ax.scatter(pc[0, :], pc[1, :], pc[2, :], marker='.', color='g', s=50)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D points from depth image', fontsize=15)
    set_axes_equal(ax)
                  
    plt.show()
