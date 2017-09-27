# First question for P1
import csv
import numpy as np
from scipy.stats import multivariate_normal
from loadParametersP1 import getData as parameters



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
        difference = 2*threshold

        while abs(difference) > threshold and num_step < 10000:
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


def evaluate(value, f):
    return f(value)


# Quad bowl function

def function_quad(A, b):
    def function_quad_sampler(x):
        transpose_x = np.transpose(x)
        left = 0.5 * np.dot(np.dot(transpose_x, A), x)
        right = np.dot(transpose_x, b)
        return left - right
    return function_quad_sampler


def function_quad_der(A,b):
    def function_quad_der_sampler(x):
        return np.dot(A, x) - b
    return function_quad_der_sampler


# Negative gaussian function

def function_gaussian(mean, cov):
    def function_gaussian_sampler(x):
        constant = 10000.0
        return constant * np.negative(multivariate_normal.pdf(x, mean, cov))
    return function_gaussian_sampler

def function_gaussian_der(mean, cov):
    def function_gaussian_der_sampler(x):
        inverse_cov = np.linalg.inv(cov)
        constant = 10000.0
        return constant*multivariate_normal.pdf(x, mean, cov) * np.dot(inverse_cov, x - mean)
    return function_gaussian_der_sampler



# Initialization
par = parameters()

gaussMean = par[0]
gaussCov = par[1]
quadBowlA = par[2]
quadBowlb = par[3]

func_quad = function_quad(quadBowlA, quadBowlb)
func_quad_der = function_quad_der(quadBowlA, quadBowlb)
func_gauss = function_gaussian(gaussMean, gaussCov)
func_gauss_der = function_gaussian_der(gaussMean, gaussCov)



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


def run_quad(initial_guess, step_size, threshold):
    result_array = []
    print "Quad stuff"
    for i in initial_guess:
        for j in step_size:
            for k in threshold:
                print "Initial guess: ", i, " Step size: ", j, " Epsilon: ", k
                value, steps = batch_gradient_descent(i, j, k, func_quad, func_quad_der, False)
                print "Steps:", steps
                print "Min value", value
                result_array.append((i, j, k, steps, value, evaluate(value, func_quad_der)))
    return result_array



# Testing

if __name__ == '__main__':

    initial_guess = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([100.0, 100.0])]
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
