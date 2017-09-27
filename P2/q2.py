import numpy as np
from q1 import compute_X
from q1 import max_like_vector
from loadFittingDataP2 import getData as parameters

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

def der_finite_diff(step, f, x):
	r = np.zeros(x.shape)
	for i in range(x.size):
		d = np.zeros(x.shape)
		d[i] = float(step)
		r[i] = (f(x+d)-f(x))/float(step)
	return r

par = parameters(False)
x = par[0]
y = par[1]
m = 0
b = np.ones(m+1)

if __name__ == '__main__':
    
    X = compute_X(x,m)
    l = loss_function(X,y)
    l_der = loss_function_der(X,y)

    print "sse:", l(b)
    print "closed derviative:", l_der(b)
    print "num approx derivative:", der_finite_diff(0.00001,l,b)
