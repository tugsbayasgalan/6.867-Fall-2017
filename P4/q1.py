import numpy as np
import math
import matplotlib.pyplot as plt
import pylab as pl
import lassoData as ld

def batch_gradient_descent(init, step_size, threshold, f, f_der, gauss):
	"""
	init - initial guess
	step_size - step size for the GD
	f - objective function
	f_der - derivative of the objective function

	return x_min

	"""

	if gauss:
		current_value = np.copy(init)
		gradient = f_der(current_value)
		num_step = 0
		difference = 100000000.0

		while abs(difference) > threshold and num_step < 1000000:
			f_now = f(current_value)
			current_value -= step_size * gradient
			f_later = f(current_value)
			gradient = f_der(current_value)
			difference = f_now - f_later
			num_step += 1

		return (current_value, num_step)


	else:

		current_value = np.copy(init)
		gradient = f_der(current_value)
		num_step = 0

		while np.linalg.norm(gradient) >= threshold:
			current_value -= step_size * gradient
			gradient = f_der(current_value)
			num_step += 1

		return (current_value, num_step)



def compute_sinX(X,M):
	r = np.zeros((X.size,M))
	for i in range(X.size):
		r[i,0] = X[i]
		for j in range(1,M):
			r[i,j] = math.sin(0.4*j*math.pi*X[i])
	return r	

def lasso_func_gen(X,y,l):
	def lasso_func(w):
		ideal = np.dot(X,w)
		loss = ideal - y
		loss_s = loss**2
		loss_sum_avg = np.sum(loss_s)/float(y.size)
		f = loss_sum_avg + l*np.sum(np.absolute(w))
		return f
	return lasso_func

def lasso_der_gen(X,y,l):
	def lasso_der(w):
		#print w.shape  	
		loss = np.dot(X,w)-y
		#print np.transpose(X).shape
		#print loss.shape
		f = 2.0*np.dot(np.transpose(X),loss)/float(y.size) + l*np.sign(w)
		return f
		#pass
	return lasso_der

def der_finite_diff(step, f, x):
	r = np.zeros(x.shape)
	for i in range(x.size):
		d = np.zeros(x.shape)
		d[i] = float(step)
		r[i] = (f(x+d)-f(x))/float(step)
	return r


x,y = ld.lassoTrainData()
m = 13
l = 0.1
X = compute_sinX(x,m)
loss = lasso_func_gen(X,y,l)
loss_der = lasso_der_gen(X,y,l)
B = pl.loadtxt("lasso_true_w.txt")
f = np.zeros(y.size)
for i in range(y.size):
	f[i] = y[i,0]
y = f

if __name__ == '__main__':

	b,step_count = batch_gradient_descent(l*np.ones((m,1)),0.0001,1e-10,loss,loss_der,True)
	print X
	print y
	print "b:", b
	print "closed b:", B
	print "steps:", step_count
	print "loss:", loss(b)
	print "loss_der:", loss_der(b)
