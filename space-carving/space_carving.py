"""
CS 6384 Homework 3 Programming
Space Carving
"""

import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


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


#TODO: implement this function for space carving
# Follow lecture 13
# mask1, mask2 are masks of the cracker box on two images
# RT1, RT2 are camera poses with shape (4, 4)
# K is the camera intrinsic matrix with shape (3, 3)
def space_carving(mask1, mask2, RT1, RT2, K):

    # define a voxel space with size N
    N = 20
    voxels = np.zeros((N, N, N), dtype=np.float32)
    
    # the range of the voxel space
    # bottom is the lower limit of the voxel space for each dimension xyz
    bottom = np.array([-0.2, -0.2, 0], dtype=np.float32)
    # top is the upper limit of the voxel space for each dimension xyz
    top = np.array([0.2, 0.2, 0.2], dtype=np.float32)
    step = (top - bottom) / N
    
    # implement the space carving algorithm to fill the voxels

    p1 = K@RT1[:-1,:]
    p2 = K@RT2[:-1,:]
    count=0
    kt=0
    for z in np.arange(bottom[2],top[2],step[2]):
        j=0
        for y in np.arange(bottom[1],top[1],step[1]):
            i=0
            for x in np.arange(bottom[0],top[0],step[0]):
                point = np.array([x,y,z,1])
                x2dhomo1 = p1@point
                y1,x1 = (x2dhomo1//x2dhomo1[2])[:-1]
                x2dhomo2 = p2@point
                y2,x2 = (x2dhomo2//x2dhomo2[2])[:-1]

                if 0<=x2<480 and 0<=x1<480 and 0<=y1<640 and 0<=y2<640:
                    if mask1[int(x1),int(y1)]>0 and mask2[int(x2),int(y2)]>0:
                        count+=1
                        voxels[i][j][kt]=1
                i+=1
            j+=1
        kt+=1

    print(count)


    
    return voxels
    
    
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
    im2, depth2, mask2, meta2 = read_data(8)
    
    # intrinsic matrix
    intrinsic_matrix = meta1['intrinsic_matrix']
    print('intrinsic_matrix')
    print(intrinsic_matrix)
    
    # camera poses
    RT1 = meta1['camera_pose']
    RT2 = meta2['camera_pose']
    print(RT1.shape, RT2.shape)
    
    # TODO: implement this function for space carving
    voxels = space_carving(mask1, mask2, RT1, RT2, intrinsic_matrix)
    
    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image 1
    ax = fig.add_subplot(2, 3, 1)
    plt.imshow(im1[:, :, (2, 1, 0)])
    ax.set_title('image 1', fontsize=15)
    
    # show mask 1
    ax = fig.add_subplot(2, 3, 2)
    plt.imshow(mask1)
    ax.set_title('mask 1', fontsize=15)
    
    # show RGB image 2    
    ax = fig.add_subplot(2, 3, 4)
    plt.imshow(im2[:, :, (2, 1, 0)])
    ax.set_title('image 2', fontsize=15)
    
    # show mask 2
    ax = fig.add_subplot(2, 3, 5)
    plt.imshow(mask1)
    ax.set_title('mask 2', fontsize=15)
    
    # show voxels for space carving
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.voxels(voxels, facecolors='r', edgecolor='k')
    set_axes_equal(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('voxel space', fontsize=15)    
    
    plt.show()
