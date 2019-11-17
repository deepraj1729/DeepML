# A Simple Linear Threshold gate to implement AND gate

"""
Truth table for AND gate
  A  B    Y = A.B
  0  0     0
  0  1     0
  1  0     0
  1  1     1
# enter two inputs 0 or 1 (press enter after typing the first input and then type the 2nd input)
"""
import numpy as np
import math


# sigmoid function always return values between 0 and 1

def sigmoid(a):
    y = 1 / (1 + (math.exp(-a)))

    if y >= 0.5:
        return 1
    else:
        return 0

try:
    theta = np.array([
        [-30, 20, 20]        # weights of AND gate
    ])
    # 1*3 dimensional matrix

    inputx1 = int(input("Enter input1 (0 or 1): " ))  # enter either 0 or 1
    if (inputx1 < 0 or inputx1 > 1):
        print("Check input values again. Has to be either (0 or 1)")
        inputx1 = int(input("Enter input1 (0 or 1): " ))
    else:
        inputx2 = int(input("Enter input2 (0 or 1): " ))  # enter either 0 or 1

    if (inputx2 < 0 or inputx2 > 1):
        print("Check input values again. Has to be either (0 or 1)")
        inputx2 = int(input("Enter input2 (0 or 1): " ))  # enter either 0 or 1
    
    X = np.array([
        [1], [inputx1], [inputx2]
    ])

    # Here the first element is called biased unit and is always 1

    # X is a 3*1 dimensional matrix

    print("\nInput values: {} {} ".format(inputx1, inputx2))

    Z = np.dot(theta, X)  # matrix multiplication (1*1 dimensional matrix)
    Y = sigmoid(Z)
    print("Output = ", Y)     # output

except Exception as e:
    print("Ooops!Something went wrong!", e)

# End of program