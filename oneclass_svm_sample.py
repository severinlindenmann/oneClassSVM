from sklearn.datasets import make_classification
from numpy import quantile, where
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import OneClassSVM

def generate_example_data():
    # Generate the example data
    x, _ = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, random_state=13)

    # Create a scatter plot of the data points
    fig, ax = plt.subplots()
    scatter = ax.scatter(x[:, 0], x[:, 1], c=_)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter plot of example data')
    plt.colorbar(scatter)

    return x, _ , fig


def create_oneclasssvm_demo(x, _, kernel, nu, gamma, degree, score):
    svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, degree=degree)

    red = svm.fit_predict(x)
    scores = svm.score_samples(x)

    thresh = quantile(scores, score)
    index = where(scores<=thresh)
    values = x[index]

    fig, ax = plt.subplots()
    ax.scatter(x[:,0], x[:,1], c=_)
    ax.scatter(values[:,0], values[:,1], color='r')

    return fig, svm


def create_oneclasssvm_2d_countour_demo(svm, x, _):
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
                     

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the scatter plot of the data points
    fig, ax = plt.subplots()
    ax.scatter(x[:,0], x[:,1])

    # Plot the decision contour
    ax.contourf(xx, yy, Z, alpha=0.2)

    return fig
