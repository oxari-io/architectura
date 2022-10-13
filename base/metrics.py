import pandas as pd
import numpy as np

def calculate_smape(actual, predicted) -> float:
  
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray), 
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual),
        np.array(predicted)
  
    return round(
        np.mean(
            np.abs(predicted - actual) / 
            ((np.abs(predicted) + np.abs(actual))/2)
        )*100, 2
    )

def smape(a, f):
    """
    a --> actual (y_true)
    f --> forecast (y_pred)
    """
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)  

def mape(A, F):
    tmp = np.abs(A - F) / np.abs(A)
    len_ = len(tmp)
    return 100 * np.sum(tmp) / len_


def adjusted_r_squared(X, Y, r2):
    '''
    Returns a computed Adjusted R-Squared Coefficient.

    Parameters
    ----------
    X : Pandas DataFrame
        A pandas DataFrame including all the independant variables. Could be a series if there is only one predictor.

    Y : Pandas DataFrame or Series
        Labels or response variables 'Y'.

    r2 : float
        R-Squared Coefficient
    '''
    n = len(Y)
    p = X.shape[1]
    adj_r = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)

    return adj_r
