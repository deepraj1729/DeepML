import numpy as np
import matplotlib.pyplot as plt
from DeepNet.layers import Dense,Input,Flatten
from DeepNet.activation import sigmoid,sigmoid_p,softmax,relu

class Perceptron():
    def __init__(self):
        print("\nDeepNet is initialized....\n")
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

    def check_acc(self,x):
        return np.round(x)

    def train__acc(self):
        return (self.acc_count/self.epochs)

    def train(self,data_X,data_Y,epochs):
        try:
            self.epochs = epochs
            for i in range(1,self.epochs+1): 
                ri = np.random.randint(len(data_X))
                point = data_X[ri]
                z = point[0] * self.w1 + point[1] * self.w2 + self.b
                
                # pred = self.sigmoid(z)
                pred = sigmoid(z)
                target = data_Y[ri]

                if self.check_acc(pred) == target:
                    self.acc_count+=1
                self.cost = np.square(pred - target)
                
                if i% 1000 == 0:
                    print("epoch: {}   train_acc = {}   loss/cost = {}".format(i,self.train__acc(),self.cost))

                self.costs.append(self.cost)
                self.counter.append(i)

                dcost_dpred = 2*(pred-target)
                # dpred_dz = self.sigmoid_p(z)
                dpred_dz = sigmoid_p(z)
                
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
        except:
            print("Oops something went wrong while training...\nCheck the input shape again")
        
    def pred(self,x):
        try:
            print("\n\n-----------------------Output--------------------")
            for i in range(len(x)):
                # conf = self.sigmoid(self.w1*x[i][0] + self.w2 *x[i][1] + self.b) 
                conf = sigmoid(self.w1*x[i][0] + self.w2 *x[i][1] + self.b) 
                val = self.check_acc(conf)
                if val == self.label[0]:
                    print("{}. x[0] = {}  x[1] = {}   predicted label: {}    confidence = {} \n".format(i+1,x[i][0],x[i][1],self.label[0],conf))
                else:
                    print("{}. x[0] = {}  x[1] = {}   predicted label: {}    confidence = {} \n".format(i+1,x[i][0],x[i][1],self.label[1],conf))
        except:
            print("Oops something went wrong...\nCheck the input shape again")

    def visualize(self):
        try:
            plt.plot(self.costs)
            plt.xlabel("Epochs")
            plt.ylabel("Cost/loss")
            plt.title("Visualisation of the optimization of the cost/loss function")
            plt.savefig('train_results/train_loss.png')
            plt.show()
        except:
            print("Oops something went wrong...\nCheck if matplotlib is installed or not")

        

