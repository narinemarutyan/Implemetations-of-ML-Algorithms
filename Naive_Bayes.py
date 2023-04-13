import numpy as np
from scipy.stats import norm

class NaiveBayes:
    def __init__(self):
        self.means = []
        self.variances = []
        self.priors = []
        
    def fit(self, X, y):
        for c in np.unique(y):
            X_c = X[y==c]
            self.means.append(X_c.mean(axis=0))
            self.variances.append(X_c.var(axis=0))
            self.priors.append(len(X_c)/len(X))
            
        self.means = np.array(self.means)
        self.variances = np.array(self.variances)
        self.priors = np.array(self.priors)

    def predict(self, X):
        y_pred = []
        for x in X:
            y = self.pdf(x, self.means, self.variances)
            y = np.log(y).sum(axis=1)
            y += np.log(self.priors)
            y_pred.append(np.argmax(y))
        return np.array(y_pred)

    def score(self, X, y):
        return (self.predict(X)==y).sum() / len(y)
        
    def pdf(self, x, mean, var):
        return norm.pdf(x, mean, var)

