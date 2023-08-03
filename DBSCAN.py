class DBSCAN:s
    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts
        
    def fit_predict(self, X):
        self.clusters = np.full(X.shape[0], -1)
        distances = pairwise_distances(X)
        self.neighbor_mtrx = distances>self.eps
        self.neighbor_counter = self.neighbor_mtrx.sum(axis=0)
        
        cluster_id = 0
        for i in range(len(X)):
            if self.neighbor_counter[i]>self.min_pts and self.clusters[i]==-1:
                self.clusters[i]=cluster_id
                self.build_cluster(i, cluster_id)
                cluster_id += 1
                    
        return self.clusters
    
    def build_cluster(self, idx, cluster_id):
        for neighbor in np.where(self.neighbor_mtrx[idx])[0]:
            if self.clusters[neighbor]==-1:    
                if self.neighbor_counter[neighbor]>self.min_pts:
                    self.clusters[neighbor]=cluster_id
                    self.build_cluster(neighbor, cluster_id)
