# A Simple Linear Threshold gate to implement NOT gate

"""
  Truth table for NOT gate:
  A       Y = A'
  ------------------------
  0         1
  1         0
  Enter one input (0 or 1)
"""

import numpy as np
import math

#sigmoid function always returns value in the range 0 and 1
def sigmoid(a):
    y = 1/(1+(math.exp(-a)))
    if y >= 0.5:
      return 1
    else:
      return 0


try:

    theta = np.array([      # weights of NOT gate
        [10], [-20]
    ])

    inputx = int(input("Enter input(0 or 1): "))  # enter either 0 or 1

    if inputx < 0 or inputx > 1:   # condition for inputs other than 0 or 1
        print("Check input values again. Has to be either (0 or 1)")
        inputx = int(input())  # enter either 0 or 1

    
    X = np.array([            # inputs of layer 1
        [1, inputx]
    ])

    print("\nInput = {}".format(inputx))


    Z = np.dot(X, theta)

    Y = sigmoid(Z)
    print("Output = {} ".format(Y))     # output


except Exception as e:
    print("Ooops!Something went wrong!", e)