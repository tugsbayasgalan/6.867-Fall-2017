import numpy as np
import math
from loadFittingDataP2 import getData as parameters
import matplotlib.pyplot as plt
import pylab as pl

def compute_cosX(X,M):
	r = np.zeros((X.size,M))
	for i in range(X.size):
		t = 1
		for j in range(1,M+1):
			r[i,j-1] = math.cos(j*math.pi*X[i])
	return r	

def max_like_vector(X,Y,M):
	r = compute_cosX(X,M)
	return np.dot(np.linalg.inv(np.dot(np.transpose(r),r)),np.dot(np.transpose(r),Y))

par = parameters(False)
X = par[0]
Y = par[1]

# Testing
# TODO maybe trying different initial values ??
if __name__ == '__main__':
	#print gaussMean
	m = 8

	r = max_like_vector(X,Y,m)
	print r
	xnew = np.linspace(X.min(),X.max(),300) #300 represents number of points to make between T.min and T.max
	y = np.dot(compute_cosX(xnew,m),r)
	plt.plot(X,Y,'o',xnew,y,'k')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()
