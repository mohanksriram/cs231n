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
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in range(num_train):
    # Calculate initial score
    f = X[i].dot(W)
    # To maintain numerical stability
    f -= np.max(f)
    # Calculate the probability by treating weighted outputs as logarithmic values
    p = np.exp(f[y[i]])/np.sum(np.exp(f))
    loss += np.sum(-np.log(p))

    for j in range(num_classes):
      p = np.exp(f[j])/np.sum(np.exp(f))
      # Gradient calculation.
      if j == y[i]:
        dW[:, j] += (-1 + p) * X[i]
      else:
        dW[:, j] += p * X[i]    

  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W) 
  dW += 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  
  # Pick all the matching scores
  matching_scores = scores[np.arange(len(y)),y]
  
  # Calculate the sofmax function
  exp_scores = np.exp(matching_scores)
  exp_sum = np.sum(np.exp(scores), axis = 1)
  softmax_output = exp_scores/exp_sum

  # Calculate the cross-entropy loss
  loss = np.sum(-np.log(softmax_output))

  # Normalize and Regularize  
  loss /= num_train
  loss += reg*np.sum(W*W)

  sub_terms = np.exp(scores)
  sum_term = np.sum(sub_terms, axis = 1)

  # Magic to obtain NXC
  p = ((sub_terms.T)/sum_term).T

  p[np.arange(len(y)), y] -= 1

  dW = X.T.dot(p)

  dW /= num_train
  dW += 2*reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

