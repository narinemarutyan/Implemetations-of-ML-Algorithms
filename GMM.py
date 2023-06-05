from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import multivariate_normal, norm

class GMM:
    def __init__(self, k, method='random_mean_std', max_iter=300, tol=1e-6):
        self.k = k 
        self.method = method
        self.max_iter = max_iter
        self.tol=tol
    
    def init_centers(self, X):
        if self.method == 'random_mean': # fix me
            kmeans = KMeans(n_clusters = self.k)
            mean_arr = np.random.rand(self.k, X.shape[1])
            kmeans.cluster_centers_ = mean_arr
            clusters = kmeans.predict(X)
            cov_arr = []
            pi_arr = []
            for i in range(self.k):
                X_i = X[clusters==i]
                cov_arr.append(np.cov(X_i.T))
                pi_arr.append(X_i.shape[0]/X.shape[0])
            return mean_arr, np.array(cov_arr), np.array(pi_arr)
        
        if self.method == 'random_mean_std':
            mean_arr = np.random.rand(self.k, X.shape[1])
            
            cov_arr = []
            for k in range(self.k):
                cov_mtrx = np.random.rand(X.shape[1], X.shape[1])
                cov_arr.append(cov_mtrx.dot(cov_mtrx.T))
            
            pi_arr = np.random.rand(self.k)
            pi_arr = pi_arr/pi_arr.sum()
            return mean_arr, np.array(cov_arr), pi_arr
        
        if self.method == 'k-means':
            # n - number of datapoints
            # m - number of features
            # k - number of clusters
            # mean_arr.shape == k x m
            # cov_arr.shape == k x m x m
            # pi_arr.shape == 1 x k
            kmeans = KMeans(n_clusters = self.k)
            kmeans.fit(X)
            clusters = kmeans.predict(X)
            mean_arr = kmeans.cluster_centers_
            cov_arr = []
            pi_arr = []
            for i in range(self.k):
                X_i = X[clusters==i]
                cov_arr.append(np.cov(X_i.T))
                pi_arr.append(X_i.shape[0]/X.shape[0])
            return mean_arr, np.array(cov_arr), np.array(pi_arr)
    
        if self.method == 'random_divide':
            idx = np.random.choice(self.k, size=X.shape[0])
            mean_arr = np.zeros((self.k, X.shape[1]))
            cov_arr = np.zeros((self.k, X.shape[1], X.shape[1]))
            pi_arr = np.zeros(self.k)
            for i in range(self.k):
                X_i = X[idx==i]
                mean_arr[i] = X_i.mean(axis=0)
                cov_arr[i] = np.cov(X_i.T)
                pi_arr[i] = X_i.shape[0]/X.shape[0]
            return mean_arr, cov_arr, pi_arr

        if self.method == 'random_gammas':
            gamma_mtrx = np.random.rand(X.shape[0], self.k)
            gamma_mtrx = gamma_mtrx/gamma_mtrx.sum(axis=1)[:, np.newaxis]
            return self.maximization(X, gamma_mtrx)

    def fit(self, X):
        self.mean_arr, self.cov_arr, self.pi_arr = self.init_centers(X)
        prev_loss = float('inf')
        for i in range(self.max_iter):
            gamma_mtrx = self.expectation(X)
            mean_arr, cov_arr, pi_arr = self.maximization(X, gamma_mtrx)
            loss = self.loss(X, mean_arr, cov_arr, pi_arr, gamma_mtrx)
            if abs(loss - prev_loss) < self.tol:
                break
            prev_loss = loss
            self.mean_arr = mean_arr
            self.cov_arr = cov_arr
            self.pi_arr = pi_arr
            
    def loss(self, X, mean, cov, pi, gamma_mtrx):
        log_likelihood = 0
        for i, x in enumerate(X):
            likelihood = 0
            for j in range(self.k):
                likelihood += pi[j] * self.pdf(x, mean[j], cov[j], allow_singular=True)
            log_likelihood += np.log(likelihood)
        loss = -log_likelihood / X.shape[0]
        return loss
    
    def pdf(self, x, mean, cov):
        return multivariate_normal.pdf(x, mean, cov, allow_singular=True)
            
    def expectation(self, X):
        gamma_mtrx = np.zeros((X.shape[0], self.k))
        for i, x in enumerate(X):
            for j in range(self.k):
                gamma_mtrx[i][j] = self.pi_arr[j] * self.pdf(x, self.mean_arr[j], self.cov_arr[j])
            gamma_mtrx[i] = gamma_mtrx[i] / gamma_mtrx[i].sum()
            
        return gamma_mtrx

    def maximization(self, X, gamma_mtrx):
        N_k = gamma_mtrx.sum(axis=0)
        N_k = np.expand_dims(N_k, axis=1) 
        mean_arr = (gamma_mtrx.T @ X) / N_k
        cov_arr = []
        for j in range(self.k):
            X_j = X - mean_arr[j]
            cov_arr.append(((X_j.T * gamma_mtrx[:, j]) @ X_j) / N_k[j])
        pi_arr = N_k / X.shape[0]
        
        return mean_arr, np.array(cov_arr), pi_arr   
    
    def loss(self, X, mean, cov, pi, gamma_mtrx):
        log_likelihood = 0
        for i, x in enumerate(X):
            likelihood = 0
            for j in range(self.k):
                likelihood += pi[j] * self.pdf(x, mean[j], cov[j])
            log_likelihood += np.log(likelihood)
        loss = -log_likelihood / X.shape[0]
        return loss
        
    def predict(self, X):
        return self.expectation(X).argmax(axis=1)
    
    def predict_proba(self, X):
        # return predictions using expectation function
        return 
