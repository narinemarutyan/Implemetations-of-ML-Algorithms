import numpy as np
from scipy.stats import norm

class LDA:
    def __init__(self):
        self.means = []
        self.cov_mtrx = None
        self.priors = []
        
    def fit(self, X, y):
        self.cov_mtrx = np.zeros((X.shape[1], X.shape[1]))

        for c in np.unique(y):
            X_c = X[y==c]
            self.means.append(X_c.mean(axis=0))
            self.priors.append(len(X_c)/len(X))
            new_diff = X_c - X_c.mean(axis=0)
            self.cov_mtrx += new_diff.T @ new_diff
        
        self.cov_mtrx = self.cov_mtrx / (len(X) - len(self.means))
        self.means = np.array(self.means)
        self.priors = np.array(self.priors)
        
    def predict(self, X):
        y_pred = []
        for x in X:
            y = []
            for p, mean in zip(self.priors, self.means):
                w1 = np.linalg.inv(self.cov_mtrx) @ mean
                w0 = np.log(p) -0.5 * mean @ w1
                y.append(x @ w1 + w0)
            y_pred.append(np.argmax(y))
        return np.array(y_pred)

    def score(self, X, y):
        return (self.predict(X)==y).sum() / len(y)
        
    def pdf(self, x, mean, var):
        return norm.pdf(x, mean, var)

