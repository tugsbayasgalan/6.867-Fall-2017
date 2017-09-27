import numpy as np
from scipy.stats import multivariate_normal
from q1 import compute_X
from q1 import max_like_vector
from loadFittingDataP2 import getData as parameters


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

def run_gauss(initial_guess, step_size, threshold):
    result_array = []
    print "Gaussian stuff"
    for i in initial_guess:
        for j in step_size:
            for k in threshold:
                print "Initial guess: ", i, " Step size: ", j, " Epsilon: ", k
                value, steps = batch_gradient_descent(i, j, k, func_gauss, func_gauss_der, True)
                print "Steps:", steps
                print "Min value", value
                result_array.append((i, j, k, steps, value, evaluate(value, func_gauss_der)))
    return result_array


par = parameters(False)
x = par[0]
y = par[1]
m = 1
X = compute_X(x,m)
B = max_like_vector(x,y,m)

loss = loss_function(X,y)
loss_der = loss_function_der(X,y)

if __name__ == '__main__':

	b,step_count =  batch_gradient_descent(np.zeros(m+1),0.0001,1e-10,loss,loss_der,True)
	print "b:", b
	print "closed b:", B
	print "steps:", step_count
	print "loss:", loss(b)
	print "loss_der:", loss_der(b)

'''    initial_guess = [np.zeros(m+1), np.ones(m+1), 100*np.ones(m+1) ]
    step_size = [0.0001, 0.00001, 0.000001]
    threshold = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    value_gauss = run_gauss(initial_guess, step_size, threshold)
    value_quad = run_quad(initial_guess, step_size, threshold)

    with open("gauss.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(value_gauss)
    with open("quad.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(value_quad)
'''
