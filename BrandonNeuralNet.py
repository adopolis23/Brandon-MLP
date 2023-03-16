import pandas as pd
import numpy as np


class NeuralNet1:
    def __init__():
        print("Error required dimentions of network")

    def __init__(self, inputs, hidden, outputs):
        self.input_num = inputs
        self.hidden_num = hidden
        self.output_num = outputs

        self.setupWeights()

        print("Init Successful with " + str(self.hidden_num) + " hidden nodes.")

    #activation functions ----

    def sigmoid(self, x):
        return (1/(1 + np.exp(-x)))

    def relu(self, x):
        if x >= 0:
            return 0
        else:
            return x





    def setupWeights(self):
        #add 1 to weights col for bias
        self.w1 = np.random.rand(self.hidden_num, self.input_num+1)
        self.w0 = np.random.rand(self.output_num, self.hidden_num+1)

    def predict(self, x):

        #append 1 for bias term then turn into numpy array
        x.append(1)
        x_np = np.array(x).reshape(self.input_num+1,1)
        
        hidden_out = np.dot(self.w1, x_np)
        
        print(hidden_out)
        #code to apply sigmoid to each element in array need to refactor
        for x in hidden_out:
            x[0] = self.sigmoid(x[0])
        

        print(hidden_out)

    #helper functions -----

    def _printWeights(self):
        print("W1 Weights: \n" + str(self.w1))
        print("W0 Weights: \n" + str(self.w0))

        


         
