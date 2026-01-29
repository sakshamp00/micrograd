# This file contains the calculation for Mean Squared Error loss function.
def MSE(y_pred, y_true):
    assert len(y_pred) == len(y_true) # Ensure both lists have the same length
    return sum((yp - yt)**2 for yp, yt in zip(y_pred, y_true))