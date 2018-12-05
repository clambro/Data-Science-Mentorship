import numpy as np
import matplotlib.pyplot as plt

NUM_DIMS = 100
NUM_ITERATIONS = 20000  # Run this many steps of each algorithm
np.random.seed(2112)  # Bonus assignemnt: Figure out why I always set my random seeds to 2112
INIT_X = np.random.uniform(-1, 1, NUM_DIMS)  # Use this as your initial x to get a fair comparison between methods


def f(x):
    """This is the function we want to minimize.

    Parameters
    ----------
    x: numpy array of floats

    Returns
    -------
    f(x): float

    """

    r2 = np.sum(x**2) / len(x)
    return 2 - np.exp(-r2) - np.exp(-r2 * 100)


def grad_f(x):
    """This is the gradient of f.

    Parameters
    ----------
    x: numpy array of n floats

    Returns
    -------
    grad_f(x): numpy array of n floats

    """

    r2 = np.sum(x**2) / len(x)
    return 2 * x * r2 * (np.exp(-r2) + 100 * np.exp(-r2 * 100)) / len(x)


def grad_desc(lr=0.01):
    """Vanilla gradient descent.

    Parameters
    ----------
    lr: float
        The learning rate.

    Returns
    -------
    loss:  The value of f at each step in the optimization.

    """

    x = INIT_X
    loss = []
    for i in range(NUM_ITERATIONS):
        grad_loss = grad_f(x)
        x = x - lr * grad_loss
        loss.append(f(x))
    return loss


def grad_desc_momentum(lr=0.01, gamma=0.9):
    """Gradient descent with momentum.

    Parameters
    ----------
    lr: float
        The learning rate.
    gamma: float between 0 and 1
        The decay parameter for the momentum.

    Returns
    -------
    loss:  The value of f at each step in the optimization

    """

    mu = 0
    x = INIT_X
    loss = []
    for i in range(NUM_ITERATIONS):
        grad_loss = grad_f(x)
        mu = gamma * mu + lr * grad_loss
        x = x - mu
        loss.append(f(x))
    return loss


def rmsprop(lr=0.001, gamma=0.9, eps=1e-8):
    """The RMSProp algorithm.

    Parameters
    ----------
    lr: float
        The learning rate.
    gamma: float between 0 and 1
        The decay parameter for the square of the gradient.
    eps: float > 0
        Parameter to prevent division by zero.

    Returns
    -------
    loss:  The value of f at each step in the optimization

    """

    nu = 0
    x = INIT_X
    loss = []
    for i in range(NUM_ITERATIONS):
        grad_loss = grad_f(x)
        nu = gamma * nu + (1 - gamma) * grad_loss**2
        x = x - lr * grad_loss / np.sqrt(nu + eps)
        loss.append(f(x))
    return loss


def adam(lr=0.001, beta1=0.9, beta2=0.99, eps=1e-8):
    """The Adam algorithm.

    Parameters
    ----------
    lr: float
        The learning rate.
    beta1: float between 0 and 1
        The decay parameter for the momentum.
    beta2: float between 0 and 1
        The decay parameter for the square of the gradient.
    eps: float > 0
        Parameter to prevent division by zero.

    Returns
    -------
    loss:  The value of f at each step in the optimization

    """

    mu = 0
    nu = 0
    x = INIT_X
    loss = []
    for i in range(NUM_ITERATIONS):
        grad_loss = grad_f(x)
        mu = beta1 * mu + (1 - beta1) * grad_loss
        nu = beta2 * nu + (1 - beta2) * grad_loss**2
        mu_hat = mu / (1 - beta1**(i + 1))
        nu_hat = nu / (1 - beta2**(i + 1))
        x = x - lr * mu_hat / (np.sqrt(nu_hat) + eps)
        loss.append(f(x))
    return loss


def plot_comparison():
    """Plots a comparison of the preceding four algorithms."""
    plt.figure()
    plt.semilogy(grad_desc())
    plt.semilogy(grad_desc_momentum())
    plt.semilogy(rmsprop())
    plt.semilogy(adam())
    plt.legend(['Gradient Descent', 'Momentum', 'RMSProp', 'Adam'])
