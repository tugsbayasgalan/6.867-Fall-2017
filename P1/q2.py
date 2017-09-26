import numpy as np
from scipy.stats import multivariate_normal
from loadParametersP1 import getData as parameters

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
        return 10000.0*multivariate_normal.pdf(x, mean, cov) * np.dot(inverse_cov, x - mean)
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
if __name__ == '__main__':
    #print gaussMean
    #print gaussCov
    #print quadBowlA
    #print quadBowlb

    test_init = np.array([1.0, 1.0])
    print "Last gradient:", evaluate(value, func_quad_der)
    print "Min value:", evaluate(value, func_quad)

    #print batch_gradient_descent(test_init, 0.001, 0.001, func_gauss, func_gauss_der)
