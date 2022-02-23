"""
CS 6384 Homework 2 Programming
Implement the harris_corner() function and the non_maximum_suppression() function in this python script
Harris corner detector
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


#TODO: implement this function
# input: R is a Harris corner score matrix with shape [height, width]
# output: mask with shape [height, width] with valuse 0 and 1, where 1s indicate corners of the input image 
# idea: for each pixel, check its 8 neighborhoods in the image. If the pixel is the maximum compared to these
# 8 neighborhoods, mark it as a corner with value 1. Otherwise, mark it as non-corner with value 0
def non_maximum_suppression(R):

    mask = np.zeros(shape=R.shape)
    h,w = R.shape

    d = [
            [0,1],
            [1,0],
            [0,-1],
            [-1,0],
            [1,1],
            [1,-1],
            [-1,1],
            [-1,-1],
    ]

    count=0
    for x in range(h):
        for y in range(w):    
            flag =True
            for i,j in d:
                px,py = x+i,y+j
                if 0<=px<h and 0<=py<w:
                    if R[px][py]>=R[x][y]:
                        flag = False
                        break 
            if flag:
                count+=1
                mask[x][y]=1
    print("==== Corners Detected========")
    print("    ", count, "        ")
    return mask


#TODO: implement this function
# input: im is an RGB image with shape [height, width, 3]
# output: corner_mask with shape [height, width] with valuse 0 and 1, where 1s indicate corners of the input image
# Follow the steps in Lecture 7 slides 29-30
# You can use opencv functions and numpy functions
def harris_corner(im):

    # step 0: convert RGB to gray-scale image
    # 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
    # print(len(im),len(im[0]))
    h,w = len(im), len(im[0])
    
    grayScale = np.ones(shape=(480,640))
    for x in range(h):
        for y in range(w):
            r,g,b = im[x][y]
            # if r>g and r>b:
            #   grayScale[x][y] = 0
            grayScale[x][y] = 0.299*r+0.587*g+0.114*b
    
    
    # Printing grayscale image
    fig = plt.figure()    
    # # show RGB image
    # ax = fig.add_subplot(2, 2, 1)
    # plt.imshow(grayScale,cmap='gray')
    # plt.show()

    
    # step 1: compute image gradient using Sobel filters
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    
    # k =3
    # sobelFilter = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    # sx = sobelFilter
    # sy = sobelFilter.T
    # r = h-k+1
    # c = w-k+1
    # imGrad = np.zeros(shape=(h,k))

    # laplacian = cv2.Laplacian(grayScale,cv2.CV_64F)
    x_grad = cv2.Sobel(grayScale,cv2.CV_64F,1,0,ksize=5)
    y_grad = cv2.Sobel(grayScale,cv2.CV_64F,0,1,ksize=5)


    # plotting image gradients
    # plt.subplot(2,2,1),plt.imshow(grayScale,cmap = 'gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    # plt.show()




    # step 2: compute products of derivatives at every pixels
    xx_grad = x_grad*x_grad
    xy_grad = x_grad*y_grad
    yy_grad = y_grad*y_grad



    # step 3: compute the sums of products of derivatives at each pixel using Gaussian filter from OpenCV
    
    xx_s = cv2.GaussianBlur(xx_grad,(5,5),0)
    xy_s = cv2.GaussianBlur(xy_grad,(5,5),0)
    yy_s = cv2.GaussianBlur(yy_grad,(5,5),0)

    print(len(xx_s),len(xx_s[0]))
    print(len(xy_s),len(xy_s[0]))
    print(len(yy_s),len(yy_s[0]))
    # step 4: compute determinant and trace of the M matrix
    det = np.zeros(shape=(h,w))
    trace = np.zeros(shape=(h,w))
    R = np.zeros(shape=(h,w))
    for x in range(h):
        for y in range(w):
            # m = np.array([[xx_s[x][y]]])
            det[x][y] = xx_s[x][y]*yy_s[x][y] - xy_s[x][y]*xy_s[x][y] 
            trace[x][y] = xx_s[x][y]+yy_s[x][y]
            R[x][y] = det[x][y] - 0.05*(trace[x][y]**2)
    

    # step 5: compute R scores with k = 0.05
    k = 0.05

    
    # step 6: thresholding
    # up to now, you shall get a R score matrix with shape [height, width]
    threshold = 0.01 * R.max()
    R[R < threshold] = 0
    
    # step 7: non-maximum suppression
    #TODO implement the non_maximum_suppression function above
    corner_mask = non_maximum_suppression(R)

    return corner_mask


# main function
if __name__ == '__main__':

    # read the image in data
    # rgb image
    rgb_filename = 'data/000006-color.jpg'
    im = cv2.imread(rgb_filename)
    
    # your implementation of the harris corner detector
    corner_mask = harris_corner(im)
    
    # opencv harris corner
    img = im.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    opencv_mask = dst > 0.01 * dst.max()
        
    # visualization for your debugging
    fig = plt.figure()
        
    # show RGB image
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax.set_title('RGB image')
        
    # show our corner image
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(corner_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('our corner image')
    
    # show opencv corner image
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(im[:, :, (2, 1, 0)])
    index = np.where(opencv_mask > 0)
    plt.scatter(x=index[1], y=index[0], c='y', s=5)
    ax.set_title('opencv corner image')

    plt.show()
