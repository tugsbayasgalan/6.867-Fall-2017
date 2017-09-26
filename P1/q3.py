import numpy as np
from q1 import batch_gradient_descent
from loadFittingDataP1 import getData
# Batch Gradient Descent

def loss_function(X, y):
    def loss_function_sampler(theta):
        ideal = np.dot(X, theta)
        loss = ideal - y
        loss_squared = loss**2
        loss_sum = np.sum(loss_squared)
        return loss_sum
    return loss_function_sampler

def loss_function_der(X, y):
    def loss_function_der_sampler(theta):
        loss = np.dot(X, theta) - y
        return np.dot(np.transpose(X),loss)
    return loss_function_der_sampler

data = getData()

X = data[0]
y = data[1]

init = np.ones(10)

func_loss = loss_function(X, y)
func_loss_der = loss_function_der(X, y)

value, steps = batch_gradient_descent(init, 0.0001, 0.0000001, func_loss, func_loss_der, True)
print "Value: ", value
print "Steps: ", steps

print func_loss(value)






# Stochastic Gradient Descent
