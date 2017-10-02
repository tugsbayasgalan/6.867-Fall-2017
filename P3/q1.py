import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.linear_model import Ridge
from regressData import *

sys.path.insert(0, '../P2')

from loadFittingDataP2 import getData
from q1 import compute_X as make_basis





def min_error(M_list, l_list, X, Y, X_test, Y_test):

    errors = []

    for M in M_list:
        for l in l_list:
            ridge_weight = ridge_w(X, Y, M, l)
            error = square_error(X_test, Y_test, ridge_weight)
            errors.append(error)
            print "M: ", M, " l:", l, "Error: ", error
    return errors



def ridge_w(X, Y, M, ridge):
    phi = transform(X, M)
    phi_T = np.transpose(phi)
    phi_squared = np.dot(phi_T, phi)
    ridge_identity = ridge * np.identity(phi_squared.shape[0])
    inv_trans = np.linalg.inv(ridge_identity + phi_squared)
    inv_trans_pseudo = np.dot(inv_trans, phi_T)
    return np.dot(inv_trans_pseudo, Y)


def transform(X, M):
    return make_basis(X, M)


def true_values(X):
  return np.cos(np.pi*X) + np.cos(2*np.pi*X)



def square_error(X, Y, w):
    basis = make_basis(X, w.size -1)
    error = np.sum((basis.dot(w) - Y) ** 2)
    return error

def calculate_error(Y, Y_pred):
    pass

def plot_one(M, X, Y, true_Y, lambda_list):


    for l in lambda_list:
        weight_ridge = ridge_w(X, Y, M, l)
        #print weight_ridge
        basis_function = np.polynomial.Polynomial(weight_ridge)
        #print "M:", i, " l:", l
        pred_Y = np.apply_along_axis(basis_function, 0, X)
        #print "Predicted Y: ", pred_Y
        plt.plot(X, pred_Y, color=colormap(normalize(l)), label="lambda: " + str(l))

    no_weight_ridge = ridge_w(X, Y, M, 0)
    no_basis_function = np.polynomial.Polynomial(no_weight_ridge)
    no_pred_Y = np.apply_along_axis(no_basis_function, 0, X)
    plt.plot(X, no_pred_Y, 'b', label="$\lambda = 0$", alpha=0.8)
    plt.scatter(X, Y, c = 'g', label="Test Points", alpha=0.8)
    plt.title("Ridge Regression for $M = " + str(M) + "$")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim((-0.01, 1.01))
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(M)
    cb = plt.colorbar(scalarmappaple, ticks=[0, 1, 1.5, 2])

    plt.legend()
    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    data = getData(False)

    # X = data[0]
    # Y = data[1]
    #
    M = [3]
    lambda_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.8, 0.6]
    # # normalize = mcolors.Normalize(vmin=-2, vmax=2)
    # # colormap =cm.OrRd
    # # #plot_one(4, X, Y, true_values(X), lambda_list)
    # # plot_one(10, X, Y, true_values(X), lambda_list)
    #
    train_X, train_Y = regressAData()
    test_X, test_Y = regressBData()
    val_X, val_Y = validateData()
    print min_error(M, lambda_list, test_X, test_Y, val_X, val_Y)
