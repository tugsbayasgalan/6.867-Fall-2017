# First question for P1

import numpy as np
from scipy.stats import multivariate_normal
from loadParametersP1 import getData as parameters



def batch_gradient_descent(init, step_size, threshold, f, f_der):
    """
    init - initial guess
    step_size - step size for the GD
    f - objective function
    f_der - derivative of the objective function

    return x_min

    """

    current_value = np.copy(init)
    gradient = f_der(current_value)
    num_step = 0

    while np.linalg.norm(gradient) >= threshold:
        current_value -= step_size * gradient
        gradient = f_der(current_value)
        num_step += 1

    return (current_value, num_step)


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
        #TODO not implemented yet
        pass
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


# Testing
# TODO maybe trying different initial values ??
