# A Simple Linear Threshold Gate to implement NOR gate
"""
  Truth table for NOR gate:
  A   B    Y = (A+B)'
  ------------------------
  0   0        1
  0   1        0
  1   0        0
  1   1        0
  Enter two inputs 0 or 1 (press enter after typing the first input and then type the 2nd input)
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

    theta = np.array([      # weights of NOR gate
        [10], [-20], [-20]
    ])
    # 3*1 dimensional matrix

    inputx1 = int(input("Enter input1 (0 or 1): " ))  # enter either 0 or 1
    if (inputx1 < 0 or inputx1 > 1):
        print("Check input values again. Has to be either (0 or 1)")
        inputx1 = int(input("Enter input1 (0 or 1): " ))
    else:
        inputx2 = int(input("Enter input2 (0 or 1): " ))  # enter either 0 or 1

    if (inputx2 < 0 or inputx2 > 1):
        print("Check input values again. Has to be either (0 or 1)")
        inputx2 = int(input("Enter input2 (0 or 1): " ))  # enter either 0 or 1

    
    X = np.array([            # inputs of layer 1
        [1, inputx1, inputx2]
    ])
    # 1*3 dimensional matrix
    # Here the first element is called biased unit and is always 1

    print("\nInput values = {} {} ".format(inputx1, inputx2))

    Z = np.dot(X, theta)

    Y = sigmoid(Z)
    print("Output = ", Y)     # output


except Exception as e:
    print("Ooops!Something went wrong!", e)