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
"""
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    w = initial_w
    batch_size = 1

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad

    loss = compute_loss(y, tx, w)

    return w, loss
"""
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

