import pandas as pd
import numpy as np
from BrandonNeuralNet import NeuralNet1



nn = NeuralNet1(2, 2, 1) 

training_x = [[0, 0], [0, 1], [1, 0], [1, 0]]
training_y = [[0], [0], [0], [1]]

#nn._printWeights()

nn.predict(training_x[0])
