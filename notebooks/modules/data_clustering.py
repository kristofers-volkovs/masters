from sklearn.cluster import DBSCAN, MeanShift
import numpy as np

class DataClustering:
    @staticmethod
    def cluster_dbscan(data, **kwargs):
        dbscan = DBSCAN(**kwargs)
        return dbscan.fit_predict(data)
    
    @staticmethod
    def cluster_meanshift(data, **kwargs):
        mean_shift = MeanShift(**kwargs)
        return mean_shift.fit_predict(data)

    @staticmethod
    def cluster_cross(data, origin, origin_radius):
        cross_labels = np.array([-1] * len(data))

        # Data point distances from origin
        dists = np.sqrt(np.sum((data - origin) ** 2, axis=1))

        # Cluster of points around the origin
        origin_idxes = np.where(dists < origin_radius)[0]
        cross_labels[origin_idxes] = 0

        # Cluster of points on the x axis
        x_bools = (data[:, 0] > origin_radius) + (data[:, 0] < -origin_radius)
        x_idxes = np.where(x_bools)
        cross_labels[x_idxes] = 1

        # Cluster of points on the y axis
        y_bools = (data[:, 1] > origin_radius) + (data[:, 1] < -origin_radius)
        y_idxes = np.where(y_bools)
        cross_labels[y_idxes] = 2

        # Points that were not labeled get sorted into the origin point cluster
        cross_labels[cross_labels == -1] = 0
        
        return cross_labels
    