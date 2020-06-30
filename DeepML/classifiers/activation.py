import numpy as np

def binaryStep(x):
    if x<0:
        return 0
    else:
        return 1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_p(x):                      # sigmoid prime
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):                           # rectified linear unit
    if x>0:
        return x
    else:
        return 0

def softmax(x):
    z = np.exp(x)
    z_ = z/z.sum()
    return z_


def tanh(x):
    return (2/(1 + np.exp(-2*x))) -1


def elu(x, a):                          # exponential linear unit
    if x<0:
        return a*(np.exp(x)-1)
    else:
        return x
