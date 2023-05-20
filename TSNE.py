from sklearn.metrics import pairwise_distances
from sklearn import datasets
import numpy as np

class TSNE:
    def __init__(self, learning_rate = 0.1, tol = 1e-6, n_features = 2, maxiter = 5, momentum = 0.9):
        self.learning_rate = learning_rate
        self.tol = tol
        self.n_features = n_features
        self.maxiter = maxiter
        self.momentum = momentum
        
    def fit(self, X, sigma = 1):        

        l2_x = pairwise_distances(X) + 1e-6 
        P = np.exp((-l2_x)/(2 * sigma**2))/sum(np.exp((-l2_x)/(2 * sigma**2)))
        P = (P + P.T) / 2
        
        y_new = np.random.rand(X.shape[0], self.n_features)
        prev_y_new = y_new.copy()
        
        for i in range(self.maxiter):
            
            l2_y_new = pairwise_distances(y_new) + 1e-6 
            Q = np.linalg.inv(1 + l2_y_new)/sum(np.linalg.inv(1 + l2_y_new))
            
            vector = y_new[:, np.newaxis, :] - y_new[np.newaxis, :, :]
            gradient = 4 * np.sum(((P - Q) * vector.T * np.linalg.inv(1+l2_y_new)), axis = 1).T
            
            y_new = y_new - self.learning_rate * gradient + self.momentum * (y_new - prev_y_new)
            
            prev_y_new = y_new.copy()

        return y_new

    
    
    
# digits = datasets.load_digits(n_class = 10)
# X = digits.data

# mds = TSNE()
# mds.fit(X)




