import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Initializing the weights and bias to zeros
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iters):
            y_pred = self.predict(X)
            
            # Calculating the gradients
            dw = (X.T.dot(y_pred - y)) / len(y)
            db = (y_pred - y).mean()
            
            # Updating the weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

