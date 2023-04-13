import numpy as np

def fit_1d_linear_regression(data_vector, response_vector):

    data_vector = np.array(data_vector)
    response_vector = np.array(response_vector)
    n = data_vector.shape[0]  # number of samples
    
    X = data_vector
    if(data_vector.ndim == 1): 
        X = data_vector.reshape(-1,1)
    X = np.concatenate((np.ones((n, 1)), X), axis=1)
    
    XtX_inverse = np.linalg.inv(np.dot(X.T, X))  # pseudo inverse for non-invertible matrices
    
    return np.dot(XtX_inverse, np.dot(X.T, response_vector))

