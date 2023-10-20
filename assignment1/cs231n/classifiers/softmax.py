from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    f = np.dot(X,W) # 得到n*c的矩阵
    f_max = np.reshape(np.max(f, axis=1), (X.shape[0],1))
    prob = np.exp(f-f_max)/np.sum(np.exp(f-f_max), axis=1, keepdims=True)
    # print(prob)
    L = -np.log(np.exp(f-f_max)/np.sum(np.exp(f-f_max), axis=1, keepdims=True))
    for i in range(X.shape[0]):
      for j in range(W.shape[1]):
        if j == y[i]:
          loss += L[i,j]
          dW[:,j] += (1 - prob[i, j])*X[i]
        else:
          dW[:,j] -= prob[i,j] * X[i]
    loss /= X.shape[0]
    loss += 0.5*reg*np.sum(W*W)
    dW = -dW / X.shape[0] + reg * W

    pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    # scores是N*C矩阵，N对应N个实例，C代表每一项的得分
    scores = X.dot(W)
    # 减去最大值防止出现数值错误
    scores -= np.max(scores, axis=1, keepdims=True)
    # 求概率，用指数矩阵除以每一行的指数行和
    prob = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    # loss,只取第二维为y的元素，伟大的numpy！
    loss = -np.sum(np.log(prob[range(num_train), y]))
    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    
    # 根据循环法的公式，先给出一个掩膜矩阵，表示当j=y[i]时，-X[i]那一项
    mask_minus_ones = np.zeros_like(scores)
    mask_minus_ones[range(num_train), y] = -1
    # 另外一个矩阵，用来乘以概率，表示循环法中两种情况下的prob*X[i]
    mask_ones = np.ones_like(scores)
    # 两个矩阵相加得到最终的更新矩阵，该更新方法与SVM类似
    update = mask_ones * prob + mask_minus_ones
    dW += np.dot(X.T, update) / num_train + reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
