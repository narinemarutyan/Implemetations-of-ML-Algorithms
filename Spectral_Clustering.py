from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy as np

class Spectral:
    
    def __init__(self,sigma = 1, threshold = 0.1, k = 3):
        self.sigma = sigma
        self.threshold = threshold
        self.k = k
    
    def fit(self, X):
        laplacian_matrix = self.calc_laplacian(X)
        eigenvalues, eigenvectors = eigh(laplacian_matrix, eigvals=(0, self.k - 1))
        return eigenvectors
    
    def predict(self, feature_vectors):
        kmeans = KMeans(n_clusters=2)
        cluster_labels = kmeans.fit_predict(feature_vectors)
        return cluster_labels

  
    def calc_laplacian(self, X):
        W = np.zeros((len(X), len(X)))
        D = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                distance = np.linalg.norm(X[i] - X[j])
                w = np.exp((-(distance**2))/self.sigma)
                if i == j:
                    W[i][j] = 0
                elif w > self.threshold:
                    W[i][j] += w
                else:
                    W[i][j] = 0 
        for i in range(len(W)):
            D[i][i] += sum(W[i])
        return D - W


# X_moons, y_moons = make_moons(n_samples=1000, noise=0.08, random_state=0)

# spectral = Spectral(sigma=0.1, threshold=0.1, k=3)
# feature_vectors = spectral.fit(X_moons)
# cluster_labels = spectral.predict(feature_vectors)

# plt.scatter(X_moons[:, 0], X_moons[:, 1], c=cluster_labels, cmap='viridis')
# plt.title('Spectral Clustering Results')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

