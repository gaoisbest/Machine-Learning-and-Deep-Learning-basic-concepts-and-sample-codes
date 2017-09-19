# -*- coding: utf-8 -*-

import numpy as np

def load_data(filepath):
    """
    Load data.
    Revised codes and the data are from Machine Learning in Action.
    """
    X = []
    Y = []
    with open(file=filepath, mode='r') as input:
        for line in input.readlines():
            x1_x2_y = line.strip().split()
            X.append([x1_x2_y[0], x1_x2_y[1]])
            Y.append(int(x1_x2_y[2]))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def sigmoid(z):
    """
    sigmoid function (i.e., activation)
    compute the sigmoid of z element-wise
    """
    return 1.0 / (1.0 + np.exp(-z))

def gradient_descent(X, Y, nx, ny, m, num_iterations, alpha):
    """
    Gradient descent to train parameters.
    """
    W = np.zeros(shape=(nx, 1), dtype=np.float32) # weights initialization
    b = 0.0 # bias initialization
    for _ in range(num_iterations):
        Z = np.dot(W.T, X) + b # shape: (1, m)
        A = sigmoid(Z) # shape: (1, m)
        cost = -1.0 / m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
        print('cost:{}'.format(cost))
        # computation graph
        dZ = A - Y # The derivative of cost to A to Z. shape: (1, m)
        dW = 1.0 / m * np.dot(X, dZ.T) # The derivative of cost to A to Z to W. shape: (nx, 1)
        W -= alpha * dW # update W
        db = 1.0 / m * np.sum(dZ) # The derivative of cost to A to Z to b
        b -= alpha * db # update b
    return W, b

def plotBestFit(X, Y, w, b):
    """
    Plot src data and final model.
    Revised codes from Machine Learning in Action.
    """
    import matplotlib.pyplot as plt
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(X.shape[1]):
        if int(Y[0, i]) == 1:
            xcord1.append(X[0, i])
            ycord1.append(X[1, i])
        else:
            xcord2.append(X[0, i])
            ycord2.append(X[1, i])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c="green")
    x = np.linspace(-3, 3, 100)
    y = -(b+w[0][0] * x) / w[1][0]
    ax.plot(x, y)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


if __name__ == '__main__':
    X, Y = load_data('test_data_from_ML_in_action.txt')
    print('before X shape:{}, Y shape:{}'.format(X.shape, Y.shape))
    X = np.transpose(X)
    Y = np.reshape(Y, newshape=(1, Y.shape[0]))
    print('after X shape:{}, Y shape:{}'.format(X.shape, Y.shape))
    nx = X.shape[0]
    ny = Y.shape[0]
    m = X.shape[1]
    print('nx:%d, ny:%d, m:%d' % (nx, ny, m))
    print('first three examples:')
    print('x:{}, y:{}'.format(X[:,0:3], Y[:,0:3]))
    num_iterations = 10000
    alpha = 0.01

    w, b = gradient_descent(X, Y, nx, ny, m, num_iterations, alpha)
    print(w)
    print(b)
    plotBestFit(X, Y, w, b)
