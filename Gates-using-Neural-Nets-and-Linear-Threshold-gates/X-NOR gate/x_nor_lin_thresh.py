# A Simple Linear Threshold Gate to implement X-NOR gate

"""
  Truth table for X-NOR gate:
  A   B    Y = AB + A'B'
  ------------------------
  0   0        1
  0   1        0
  1   0        0
  1   1        1
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
    theta_A = np.array([    # weights of AND gate
        [-30], [20], [20]
    ])

    theta_B = np.array([    # weights of (NOT A) AND (NOT B) gate
        [10], [-20], [-20]
    ])

    theta_C = np.array([    # weights of OR gate
        [-10], [20], [20]
    ])

    inputx1 = int(input("Enter input1 (0 or 1): " ))  # enter either 0 or 1
    if (inputx1 < 0 or inputx1 > 1):
        print("Check input values again. Has to be either (0 or 1)")
        inputx1 = int(input("Enter input1 (0 or 1): " ))
    else:
        inputx2 = int(input("Enter input2 (0 or 1): " ))  # enter either 0 or 1

    if (inputx2 < 0 or inputx2 > 1):
        print("Check input values again. Has to be either (0 or 1)")
        inputx2 = int(input("Enter input2 (0 or 1): " ))  # enter either 0 or 1

    
    X_L1 = np.array([       # inputs of layer 1
        [1, inputx1, inputx2]
    ])

    print("\nInput values = {} {} ".format(inputx1, inputx2))

    Z1 = np.dot(X_L1, theta_A)
    Z2 = np.dot(X_L1, theta_B)

    p1 = sigmoid(Z1)
    p2 = sigmoid(Z2)

    X_L2 = np.array([   # inputs of layer 2
        [1, p1, p2]
    ])

    Z3 = np.dot(X_L2, theta_C)

    Y = sigmoid(Z3)
    print("Output = {} ".format(Y))     # output given by layer 3


except Exception as e:
    print("Ooops!Something went wrong!", e)