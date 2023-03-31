import pandas as pd
import numpy as np
from activation_functions import sigmoid, sigmoid_derivative


class NeuralNet1:
    def __init__():
        print("Error required dimentions of network")

    def __init__(self, inputs, hidden, outputs):
        self.input_num = inputs
        self.hidden_num = hidden
        self.output_num = outputs

        self.learning_rate = 0.1

        self.setupWeights()

        print("Init Successful with " + str(self.hidden_num) + " hidden nodes.")




    def setupWeights(self):
        #add 1 to weights col for bias
        self.w1 = np.random.rand(self.hidden_num, self.input_num+1)
        self.w0 = np.random.rand(self.output_num, self.hidden_num+1)

    def predict(self, input):

        #append 1 for bias term then turn into numpy array
        x = input + [1]
        #x.append(1)
        x_np = np.array(x).reshape(self.input_num+1,1)
        
        #dot product of weights 1 and inputs
        hidden_out = np.dot(self.w1, x_np)
        
        
        #code to apply sigmoid to each element in array need to refactor
        for x in hidden_out:
            x[0] = sigmoid(x[0])
        
        #add 1 to hidden out for bias
        hidden_out = np.append(hidden_out,[1])
        
        #y is dot product of w0 and hidden outputs
        y = np.dot(self.w0, hidden_out).reshape(self.output_num, 1)

        #apply sigmoid function
        for x in y:
            x[0] = sigmoid(x[0])

        #flatten array
        flatten = [element for innerList in y for element in innerList]

        return flatten
    




    def train_one(self, input, target):

        #DELTA W = lr * (target - output) * hidden-output
        #append 1 for bias term then turn into numpy array
        x = input + [1]
        #x.append(1)
        x_np = np.array(x).reshape(self.input_num+1,1)
        
        #dot product of weights 1 and inputs
        hidden_out = np.dot(self.w1, x_np)
        
        
        #code to apply sigmoid to each element in array need to refactor
        for x in hidden_out:
            x[0] = sigmoid(x[0])
        
        #add 1 to hidden out for bias
        hidden_out = np.append(hidden_out,[1])
        
        #y is dot product of w0 and hidden outputs
        y = np.dot(self.w0, hidden_out).reshape(self.output_num, 1)

        #apply sigmoid function
        for x in y:
            x[0] = sigmoid(x[0])

        #flatten array
        flat = [element for innerList in y for element in innerList]


        #calculate delta w for hidden to output 
        for output_node in range(self.output_num):
            for i in range (len(self.w0[0])-1):
                delta = self.learning_rate * (target[output_node]-flat[output_node]) * hidden_out[i]
                self.w0[output_node][i] = self.w0[output_node][i] + delta







    #helper functions -----

    def set_learning_rate(self, x):
        self.learning_rate = x


    def _printWeights(self):
        print("W1 Weights: \n" + str(self.w1))
        print("W0 Weights: \n" + str(self.w0))

        


         
