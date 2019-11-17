import tensorflow as tf 
from tensorflow import keras
from sklearn import preprocessing, model_selection
import numpy as np 
import pandas as pd

df = pd.read_csv('or.csv')

x = np.array(df.drop(['Y_values'],1), dtype= int)
y = np.array(df['Y_values'], dtype = int)
y.shape = (len(y),1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(2,)))
model.add(tf.keras.layers.Dense(4,activation = "relu"))
model.add(tf.keras.layers.Dense(300,activation = "relu"))
model.add(tf.keras.layers.Dense(2,activation = "softmax"))


model.compile(optimizer = "adam",
       loss = 'sparse_categorical_crossentropy',
       metrics=['accuracy'])

model.fit(x ,y , epochs = 100)       

x_test = np.array([
    [1,0],
    [1,1],
    [0,0],
    [0,1]
], dtype= float)

z = np.round(model.predict(x_test))

for i in z:
    if i[0]== 1:
        print("0")
    else:
        print("1")
