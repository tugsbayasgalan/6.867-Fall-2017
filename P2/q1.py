import numpy as np
from scipy.stats import multivariate_normal
from loadFittingDataP2 import getData as parameters
import matplotlib.pyplot as plt
import pylab as pl
from scipy.interpolate import spline


def compute_X(X,M):
	r = np.zeros((X.size,M+1))
	for i in range(X.size):
		t = 1
		for j in range(M+1):
			r[i,j] = t
			t *= X[i]
	return r	

def max_like_vector(X,Y,M):
	r = compute_X(X,M)
	return np.dot(np.linalg.inv(np.dot(np.transpose(r),r)),np.dot(np.transpose(r),Y))

par = parameters(False)
X = par[0]
Y = par[1]

# Testing
# TODO maybe trying different initial values ??
if __name__ == '__main__':
	#print gaussMean
	m = 10

	r = max_like_vector(X,Y,m)
	print r
	xnew = np.linspace(X.min(),X.max(),300) #300 represents number of points to make between T.min and T.max
	y = np.dot(compute_X(xnew,m),r)
	plt.plot(X,Y,'o',xnew,y,'k')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title("Linear Regression(M="+str(m)+")")
	plt.show()


