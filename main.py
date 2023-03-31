import pandas as pd
import numpy as np
from BrandonNeuralNet import NeuralNet1



nn = NeuralNet1(2, 2, 2) 

#Logical AND dataset
training_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
training_y = [[0, 0], [0, 1], [1, 0], [1, 1]]

#nn._printWeights()

print(nn.predict(training_x[1]))

nn.train_one(training_x[1], training_y[1])
nn.train_one(training_x[1], training_y[1])
nn.train_one(training_x[1], training_y[1])
nn.train_one(training_x[1], training_y[1])
nn.train_one(training_x[1], training_y[1])

print(nn.predict(training_x[1]))
