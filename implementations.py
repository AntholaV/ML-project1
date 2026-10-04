import numpy as np


def calculate_mse(e):
    """Calculate the mean squared error for an error vector e."""
    return 1 / 2 * np.mean(e**2)


def compute_loss(y, tx, w):
    """Compute the MSE loss."""
    e = y - tx.dot(w)

    return calculate_mse(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)

    return grad, err


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    w = initial_w

    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)

        # gradient w by descent update
        w = w - gamma * grad

    loss = compute_loss(y, tx, w)

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    w = initial_w

    for n_iter in range(max_iters):
        i = np.random.randint(len(y))

        y_batch = y[i:i + 1]
        tx_batch = tx[i:i + 1]

        grad, _ = compute_gradient(y_batch, tx_batch, w)
        w = w - gamma * grad

    loss = compute_loss(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """Compute the least squares solution using the normal equations.

    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N, D)

    Returns:
        w: optimal weights, numpy array of shape (D,)
        loss: MSE loss as a scalar
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    """Implement ridge regression."""

    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)

    return w, loss

def sigmoid(t):
    """Apply the sigmoid function."""
    return 1 / (1 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """Compute the logistic loss."""
    z = tx.dot(w)
    loss = np.mean(np.logaddexp(0, z) - y * z)
    return loss


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) / len(y)
    return grad


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent."""

    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        w = w - gamma * grad

    loss = compute_logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent."""

    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        grad = grad + 2 * lambda_ * w
        w = w - gamma * grad

    loss = compute_logistic_loss(y, tx, w)

    return w, loss