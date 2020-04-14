# Perceptron Network
#
#      O     Flower Type(0-Blue,1-Red)
#    /   \        w1,w2,b
#   O     O   Inputs (length,width)

import numpy as np
import pandas as pd
from DeepNet.models import Perceptron

data = pd.read_csv('Datasets/flower.csv')

x = np.array(data.drop(['Sl no.','flower label'],1))
y = np.array(data['flower label'])

mystery_flower = np.array([[4.8,1.9],[2,1.5],[7,3.9]])

model = Perceptron()
model.train(data_X=x,data_Y=y,epochs= 50000)
model.train__accuracy()
model.test(mystery_flower)
model.visualize()