import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math
from layers import Dense,Input
from activation import sigmoid,softmax,relu

class Perceptron():
    def __init__(self):
        print("\nClass Neural_net called....\n")
        print("Initializing weights and biases")
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.b = np.random.randn()
        self.l_rate = 0.2
        self.costs=[]
        self.counter = []
        self.cost = 0
        self.acc_count = 0
        self.epochs = 50000
        self.label = [0,1]
        

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_p(self,x): # derivative of sigmoid / sigmoid prime
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def check_acc(self,x):
        return np.round(x)

    def train__accuracy(self):
        return (self.acc_count/self.epochs)

    def train(self,data,epochs):
        self.epochs = epochs
        for i in range(1,self.epochs+1): 
            ri = np.random.randint(len(data))
            point = data[ri]
            z = point[0] * self.w1 + point[1] * self.w2 + self.b
            
            pred = self.sigmoid(z)
            target = point[2]

            if self.check_acc(pred) == target:
                self.acc_count+=1
            self.cost = np.square(pred - target)
            
            if i% 1000 == 0:
                print("epoch: {}   train_acc = {}   loss/cost = {}".format(i,self.train__accuracy(),self.cost))

            self.costs.append(self.cost)
            self.counter.append(i)

            dcost_dpred = 2*(pred-target)
            dpred_dz = self.sigmoid_p(z)
            
            dz_dw1 = point[0]
            dz_dw2 = point[1]
            dz_db = 1
            
            dcost_dz = dcost_dpred * dpred_dz  # Chain rule
            
            dcost_dw1 = dcost_dz * dz_dw1      # Chain rule
            dcost_dw2 = dcost_dz * dz_dw2      # Chain rule
            dcost_db = dcost_dz * dz_db        # Chain rule
            
            self.w1 = self.w1 - self.l_rate* dcost_dw1
            self.w2 = self.w2 - self.l_rate* dcost_dw2
            self.b  = self.b  - self.l_rate* dcost_db
        
    def test(self,x):
        print("\n\n-----------------------Output--------------------")
        for i in range(len(x)):
            conf = self.sigmoid(self.w1*x[i][0] + self.w2 *x[i][1] + self.b) 
            val = self.check_acc(conf)
            if val == self.label[0]:
                print("{}. x[0] = {}  x[1] = {}   predicted label: {}    confidence = {} \n".format(i+1,x[i][0],x[i][1],self.label[0],conf))
            else:
                print("{}. x[0] = {}  x[1] = {}   predicted label: {}    confidence = {} \n".format(i+1,x[i][0],x[i][1],self.label[1],conf))
    
    def visualize(self):

        plt.plot(self.costs)
        plt.xlabel("Epochs")
        plt.ylabel("Cost/loss")
        plt.title("Visualisation of the optimization of the cost/loss function")
        plt.show()
        plt.savefig('cost_func.png')

