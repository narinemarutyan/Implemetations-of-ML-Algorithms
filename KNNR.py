import numpy as np
from sklearn.metrics import r2_score

class KNN_Regressor:
    def __init__(self, n_neighbors = 3, method='euclidean', weights='uniform'):
        self.n_neighbors = n_neighbors
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def distance(self, x1,  x2):
        return np.sqrt(np.linalg.norm(x1-x2, axis=1))
    
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            distances = self.distance(x, self.X_train)
            indices = np.argsort(distances)[:self.n_neighbors]
            y_pred.append(np.mean(self.y_train[indices]))
        return np.array(y_pred)
    
    def score(self, X_test, y_test):
        return r2_score(y_test, self.predict(X_test))

