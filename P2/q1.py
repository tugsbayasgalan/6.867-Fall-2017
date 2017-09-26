import numpy as np
from scipy.stats import multivariate_normal
from loadFittingDataP2 import getData as parameters
import matplotlib.pyplot as plt
import pylab as pl


def max_like_vector(X,Y,M):
	r = np.zeros((X.size,M+1))
	for i in range(X.size):
		t = 1
		for j in range(M+1):
			r[i,j] = t
			t *= X[i]
	return (r,np.dot(np.linalg.inv(np.dot(np.transpose(r),r)),np.dot(np.transpose(r),Y)))

par = parameters(False)
X = par[0]
Y = par[1]

# Testing
# TODO maybe trying different initial values ??
if __name__ == '__main__':
    #print gaussMean
    r,m = max_like_vector(X,Y,3)
    y = np.dot(r,m)

    plt.plot(X,Y,'o',X,y,'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


