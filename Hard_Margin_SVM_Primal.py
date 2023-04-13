import numpy as np
from scipy.optimize import minimize

class SVM:
    def __init__(self):
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n = X.shape[0]
        d = X.shape[1]
        
        def objective(x):
            return np.dot(x[:-1], x[:-1])
        
        def constraint(x):
            return y * (np.dot(X, x[:-1]) + x[-1]) - 1

        w = np.zeros(d+1)
        cons = {'type': 'ineq', 'fun': constraint}
        res = minimize(objective, w, constraints=cons)

        self.w = res.x[:-1]
        self.b = res.x[-1] 
        
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    
    def score(self, X, y):
        return (self.predict(X)==y).sum() / len(y)

