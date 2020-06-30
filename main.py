import numpy as np
import pandas as pd
from DeepML.classifiers.models import Perceptron

data = pd.read_csv('datasets/iris.csv')

X = np.array(data.drop(['Sl no.','flower label'],1))
y = np.array(data['flower label'])

mystery_flower = np.array([[4.8,1.9],[2,1.5],[7,3.9]])

model = Perceptron()
model.train(data_X=X,data_Y=y,epochs= 50000)
model.pred(mystery_flower)
model.visualize()