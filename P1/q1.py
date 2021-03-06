# First question for P1
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
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
        store_steps = []
        store_grad = []

        while abs(difference) > threshold:
            f_now = f(current_value)
            current_value -= step_size * gradient
            f_later = f(current_value)
            gradient = f_der(current_value)
            #print "At step: ", num_step, " gradient was: ", np.linalg.norm(gradient)
            difference = f_now - f_later

            if num_step < 2000:
                #print "At step: ", num_step, " gradient was: ", np.linalg.norm(gradient)
                store_steps.append(num_step)
                store_grad.append(np.linalg.norm(gradient))

            num_step += 1

        return (store_steps, store_grad)


    else:

        current_value = np.copy(init)
        gradient = f_der(current_value)
        num_step = 0
        store_steps = []
        store_grad = []

        while np.linalg.norm(gradient) >= threshold:
            current_value -= step_size * gradient
            gradient = f_der(current_value)
            if num_step < 1001:
                #print "At step: ", num_step, " gradient was: ", np.linalg.norm(gradient)
                store_steps.append(num_step)
                store_grad.append(np.linalg.norm(gradient))


            num_step += 1

        return (current_value, num_step)
        #return (store_steps, store_grad)


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

        return np.negative(multivariate_normal.pdf(x, mean, cov))
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
print gaussMean
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
                #print "Initial guess: ", i, " Step size: ", j, " Epsilon: ", k
                value, steps = batch_gradient_descent(i, j, k, func_gauss, func_gauss_der, True)
                #print "Steps:", steps
                #print "Min value", value
                #mag_gradient = evaluate(value, func_gauss_der)
                #print "Gradient magnitude:", np.linalg.norm(evaluate(value, func_gauss_der))
                result_array.append((i, j, k, steps, value, np.linalg.norm(evaluate(value, func_gauss_der))))
    return result_array


def run_quad(initial_guess, step_size, threshold):
    result_array = []
    print "Quad stuff"
    for i in initial_guess:
        for j in step_size:
            for k in threshold:
                #print "Initial guess: ", i, " Step size: ", j, " Epsilon: ", k
                value, steps = batch_gradient_descent(i, j, k, func_quad, func_quad_der, False)
                #print "Steps:", steps
                #print "Min value", value
                #mag_gradient = evaluate(value, func_quad_der)
                #print "Gradient magnitude:", np.linalg.norm(evaluate(value, func_quad_der))
                result_array.append((i, j, k, steps, value, np.linalg.norm(evaluate(value, func_quad_der))))
    return result_array



# Testing

if __name__ == '__main__':

    initial_guess = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([100.0, 100.0])]
    step_size = [0.0001, 0.00001, 0.000001]
    threshold = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    #value_gauss = run_gauss(initial_guess, step_size, threshold)
    #value_quad = run_quad(initial_guess, step_size, threshold)

    # with open("gauss.csv", "wb") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(value_gauss)
    # with open("quad.csv", "wb") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(value_quad)
    #
    # value, steps = batch_gradient_descent(np.array([-5.0, -5.0]), 1e-3, 1e-5, func_gauss, func_gauss_der, True)
    # print value
    # print steps


    step_sizes = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]

    # for i in range(len(step_sizes)):
    #     store_steps, store_grad = batch_gradient_descent(np.array([1.0, 1.0]), step_sizes[i], 1e-5, func_quad, func_quad_der, False)
    #     if i % 4 == 0:
    #         color = 'r'
    #     if i % 4 == 1:
    #         color = 'b'
    #     if i % 4 == 2:
    #         color = 'g'
    #     if i % 4 == 3:
    #         color = 'k'
    #     if i % 2 == 1:
    #         plt.plot(store_steps, store_grad, color)
    #         plt.ylabel("Magnitude of Gradient")
    #         plt.xlabel("Number of steps")
    #
    # plt.title("Changing step sizes")
    # plt.show()
    # plt.savefig("image.png")

    initial_values = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([5.0, 5.0]), np.array([9.0, 9.0])]

    for i in range(len(initial_values)):
        store_steps, store_grad = batch_gradient_descent(initial_values[i], 0.0001, 1e-11, func_gauss, func_gauss_der, True)
        if i % 4 == 0:
            color = 'r'
        if i % 4 == 1:
            color = 'b'
        if i % 4 == 2:
            color = 'g'
        if i % 4 == 3:
            color = 'k'
        plt.plot(store_steps, store_grad, color)
        plt.ylabel("Magnitude of Gradient")
        plt.xlabel("Number of Steps")
    plt.title("Changing initial values")
    plt.show()
