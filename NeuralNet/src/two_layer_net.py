from turtle import ScrolledCanvas
import numpy as np
from classifier import Classifier
from layers import fc_forward, fc_backward, relu_forward, relu_backward


class TwoLayerNet(Classifier):
    """
    A neural network with two layers, using a ReLU nonlinearity on its one
    hidden layer. That is, the architecture should be:

    input -> FC layer -> ReLU layer -> FC layer -> scores
    """
    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512,
                 weight_scale=1e-3):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The weight matrices of the model will be initialized
          from a Gaussian distribution with standard deviation equal to
          weight_scale. The bias vectors of the model will always be
          initialized to zero.
        """
        #######################################################################
        # TODO: Initialize the weights and biases of a two-layer network.     #
        #######################################################################
        self.w1 = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.w2 = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.b2 = np.zeros(num_classes)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def parameters(self):
        params = None
        #######################################################################
        # TODO: Build a dict of all learnable parameters of this model.       #
        #######################################################################
        params = {
            "w1":self.w1,
            "w2":self.w2,
            "b1":self.b1,
            "b2":self.b2
            # "reg"
        }
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return params

    def forward(self, X):
        scores, cache = None, None
        #######################################################################
        # TODO: Implement the forward pass to compute classification scores   #
        # for the input data X. Store into cache any data that will be needed #
        # during the backward pass.                                           #
        #######################################################################

        # print("===X:",X.shape,self.w1.shape)
        s1,c1 = fc_forward(X,self.w1,self.b1)
        sr1,cr1 = relu_forward(s1)
        # print("===s1:",sr1.shape,self.w2.shape)
        s2,c2 = fc_forward(sr1,self.w2,self.b2)

        # scores = {
        #     "w1": s1,
        #     "c1": c1,
        #     "w2": s2,
        #     "c2": c2
        # }

        scores = s2
        cache = [c1,cr1,c2]

        # print("=====cache shape========")
        # print(list(map(len,cache)))
        
        # score1,cache1 = fc_forward(X,self.w1,self.b1)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################


        return scores, cache

    def backward(self, grad_scores, cache):
        grads = None
        #######################################################################
        # TODO: Implement the backward pass to compute gradients for all      #
        # learnable parameters of the model, storing them in the grads dict   #
        # above. The grads dict should give gradients for all parameters in   #
        # the dict returned by model.parameters().                            #
        #######################################################################

        # s1,s2 = grad_scores
        # print("=====backward cache shape========")
        # print(list(map(len,cache)))
        c1,cr1,c2 = cache

        gx2, grad_w2, grad_b2 = fc_backward(grad_scores, c2)
        gr = relu_backward(gx2,cr1)
        gx1, grad_w1, grad_b1 = fc_backward(gr, c1)

        grads={}
        grads['w2'] = grad_w2
        grads['b2'] = grad_b2
        grads['w1'] = grad_w1
        grads['b1'] = grad_b1

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        return grads
