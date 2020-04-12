# Perceptron Network
#
#      O     Flower Type(0-Blue,1-Red)
#    /   \        w1,w2,b
#   O     O   Inputs (length,width)

import numpy as np
from models import Perceptron

data = np.array([
    [1,1.5,0],
    [2,1,0],
    [1.5,1.2,0],
    [4,1.5,1],
    [4.5,2,1],
    [4.3,1.8,1],
    [3,1.3,0],
    [3.5,0.5,0],
    [5.5,2.5,1],
    [4.5,1.5,1],
    [2,0.5,0],
    [5.5,3,1],
    [1,1,0],
    [7.4,3.4,1]
])


mystery_flower = np.array([[4.8,1.9],[2,1.5],[7,3.9]])

model = Perceptron()
model.train(data=data,epochs= 70000)
model.train__accuracy()
model.test(mystery_flower)
model.visualize()