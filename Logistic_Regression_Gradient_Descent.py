class BinaryLogisticRegression:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
  
    def fit(self, X, y):
        
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # Adding a column of 1s for the intercept
        self.weights = np.zeros((X.shape[1]))  # Setting the initial weights to 0

        cost_history = []
        iters = []
        k = 0
        for i in range(self.n_iters):
            y_pred = self.sigmoid(np.dot(X, self.weights))
            
            dw = (X.T.dot(y_pred - y)) / len(y)
            self.weights -= self.learning_rate * dw
            c = self.cost(X, y, self.weights)
            k += 1
            iters.append(k)
            cost_history.append(c)

        return self.weights, cost_history, iters
            
    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.sigmoid(np.dot(X, self.weights)) > 0.5  # Making categorical
    
    def cost(self, X, y, w):   
        m = X.shape[0]

        c = 0
        for i in range(m):
            z = X[i].dot(w)
            c+= y[i]*np.log(self.sigmoid(z)) + (1-y[i])*np.log(1 - self.sigmoid(z))  # Logistic cost

        return -c/m  # final cost
    
    def sigmoid(self, z):
        
        g = 1/(1 + np.exp(-z))  # Formula for the sigmoid
        return g

    def score(self, x, y):
        y_pred = self.predict(x)
        return (y_pred == y).sum() / len(y)

