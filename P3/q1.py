import numpy as np
import sys
sys.path.insert(0, '../P2')

from loadFittingDataP2 import getData
from q1 import max_like_vector as make_basis


data = getData(False)

X = data[0]
y = data[1]

def ridge_w(X, Y, M, ridge):
    phi = transform(X, M)
    phi_T = np.transpose(phi)
    phi_squared = np.dot(phi_T, phi)
    ridge_identity = ridge * np.identity(phi_squared.shape[0])
    inv_trans = np.linalg.inv(npridge_identity + phi_squared)
    inv_trans_pseudo = np.dot(inv_trans, phi_T)
    return np.dot(inv_trans_pseudo, Y)


def transform(X, M):
    return make_basis(X, M)

def plot_one(M, X, Y, lambda_list):

    for i in M:
        for l in lambda_list:
            weight_ridge = ridge_w(X, Y, i, l)
            basis_function = np.polynomial.Polynomial(weight_ridge)




if __name__ == '__main__':

    M = [4 , 6, 8, 10]
    lambda_list = []
