import matplotlib.pyplot as plt
import numpy as np
from regressData import *
from q1 import ridge_w
sys.path.insert(0, '../P2')
print sys
from q1 import compute_X as make_basis

A_data = regressAData()
B_data = regressBData()
v_data = validateData()



if __name == '__main__':

    A_data = regressAData()
    B_data = regressBData()
    v_data = validateData()

    trainX, trainY = A_data
    testX, testY = B_data
    valX, valY = v_data

    M = [4 , 6, 8, 10]
    lambda_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]

    result = ridge_w(trainX, trainY, 4, 1e-3)
    print result
