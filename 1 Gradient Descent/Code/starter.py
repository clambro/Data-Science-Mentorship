import numpy as np


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


def grad_desc():
    pass


def grad_desc_momentum():
    pass


def rmsprop():
    pass


def adam():
    pass


def plot_comparison():
    pass
