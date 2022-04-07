from cv2 import divide
import numpy as np


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array of shape (N, Din) giving input data
    - w: A numpy array of shape (Din, Dout) giving weights
    - b: A numpy array of shape (Dout,) giving biases

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################
    # N = x.shape[0]
    # x_row = x.reshape(N, -1)
    # out = x_row.dot(w) + b
    
    n,din = x.shape
    out = (x@w) + np.repeat(b[None],n,axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(grad_out, cache):
    """
    Computes the backward pass for a fully-connected layer.

    Inputs:
    - grad_out: Numpy array of shape (N, Dout) giving upstream gradients
    - cache: Tuple of:
      - x: A numpy array of shape (N, Din) giving input data
      - w: A numpy array of shape (Din, Dout) giving weights
      - b: A numpy array of shape (Dout,) giving biases

    Returns a tuple of downstream gradients:
    - grad_x: A numpy array of shape (N, Din) of gradient with respect to x
    - grad_w: A numpy array of shape (Din, Dout) of gradient with respect to w
    - grad_b: A numpy array of shape (Dout,) of gradient with respect to b
    """
    x, w, b = cache
    grad_x, grad_w, grad_b = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for the fully-connected layer         #
    ###########################################################################

    N = x.shape[0]
    x_row = x.reshape(N, -1)
    grad_x = grad_out.dot(w.T)
    grad_x = grad_x.reshape(x.shape)
    grad_w = x_row.T.dot(grad_out)
    grad_b = np.sum(grad_out, axis=0)

#   Running slow probably because of @ operator
    # n,dout = grad_out.shape
    # grad_x = grad_out@w.T
    # grad_w = x.T@grad_out
    # grad_b = (np.ones(n).T)@grad_out
    # grad_b = grad_b.T
    # grad_b = np.sum(grad_out,axis=0)
    # print("Grad_b shape in Fc backward [expected (dout,)]: ", grad_b.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


    return grad_x, grad_w, grad_b


def relu_forward(x):
    """
    Computes the forward pass for the Rectified Linear Unit (ReLU) nonlinearity

    Input:
    - x: A numpy array of inputs, of any shape

    Returns a tuple of:
    - out: A numpy array of outputs, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    out = np.maximum(x,0.0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(grad_out, cache):
    """
    Computes the backward pass for a Rectified Linear Unit (ReLU) nonlinearity

    Input:
    - grad_out: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - grad_x: Gradient with respect to x
    """
    grad_x, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    # grad_x = np.minimum(1.0, np.ceil(np.maximum(x,0.0)))*grad_out

    grad_x = grad_out
    grad_x[x<=0] = 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return grad_x


def l2_loss(x, y):
    """
    Computes the loss and gradient of L2 loss.

    loss = 0.5 * sum_i (x_i - y_i)**2 / N

    Inputs:
    - x: Input data, of shape (N, D)
    - y: Output data, of shape (N, D)

    Returns a tuple of:
    - loss: Scalar giving the loss
    - grad_x: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    diff = x - y
    loss = 0.5 * np.sum(diff * diff) / N
    grad_x = diff / N
    return loss, grad_x


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax (cross-entropy) loss function.

    Inputs:
    - x: Numpy array of shape (N, C) giving predicted class scores, where
      x[i, c] gives the predicted score for class c on input sample i
    - y: Numpy array of shape (N,) giving ground-truth labels, where
      y[i] = c means that input sample i has ground truth label c, where
      0 <= c < C.

    Returns a tuple of:
    - loss: Scalar giving the loss
    - grad_x: Numpy array of shape (N, C) giving the gradient of the loss with
      with respect to x
    """

    loss, grad_x = None, None
    ###########################################################################
    # TODO: Implement softmax loss                                            #
    ###########################################################################

    # Slow execution because of looping by enumeration .... 

    # # print("============")
    # # print("shape y: ",y.shape,y )
    # # print("shape x: ",x.shape )
    # # print("============")
    # # divide = np.sum(np.exp(x))
    # batch = x
    # nBatches,nClasses = x.shape
    # loss = 0
    # grad_x = np.zeros(shape=(nBatches,nClasses),dtype=float)
    # p = np.zeros(nClasses,dtype=float)

    # for i,x in enumerate(batch):
    #   # print("============")
    #   # print("shape x: ",i,x.shape,x )
    #   # print("============")
    #   m = np.max(x)
    #   p = np.exp(x-m)/np.sum(np.exp(x-m))
    #   loss += -np.sum(np.log(p[y[i]])) 
    #   # print("los :",loss)
    #   # print("============")
    #   # print("shape p: ",p.shape,i )
    #   # print("============")
    #   delta = np.zeros(nClasses,dtype=float)
    #   delta[y[i]]=1
    #   grad_x[i] = p - delta
    # loss = loss/nBatches
    # grad_x = grad_x/nBatches
    # # print(loss)
    # # print("Grad X: ",grad_x)


    # clean code and faster

    p = np.exp(x - np.max(x, axis=1, keepdims=True)) 
    p /= np.sum(p, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(p[np.arange(N), y])) / N
    grad_x = p.copy()
    grad_x[np.arange(N), y] -= 1
    grad_x /= N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, grad_x


def l2_regularization(w, reg):
    """
    Computes loss and gradient for L2 regularization of a weight matrix:

    loss = (reg / 2) * sum_i w_i^2

    Where the sum ranges over all elements of w.

    Inputs:
    - w: Numpy array of any shape
    - reg: float giving the regularization strength

    Returns:
    """
    loss, grad_w = None, None
    ###########################################################################
    # TODO: Implement L2 regularization.                                      #
    ###########################################################################

    loss = (reg/2)*np.sum((w*w))
    grad_w = reg*w
    # print("=================")
    # print(loss)
    # print(grad_w)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, grad_w
