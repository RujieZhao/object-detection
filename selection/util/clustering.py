from numpy import unique
from numpy import where, bool_
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification,make_blobs
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
from detectron2.utils.visualizer import Visualizer
import hdbscan

# initialize the data set we'll work with
# X, _ = make_classification(
#     n_samples=1000,
#     n_features=2,
#     n_informative=2,
#     n_redundant=0,
#     n_clusters_per_class=1,
#     random_state=4
# )
# print(X.shape,type(X))
# define the models
# affinity_model = AffinityPropagation(damping=0.7)
agglomerative_model = AgglomerativeClustering(n_clusters=None,distance_threshold=3) #n_clusters=3
# birch_model = Birch(threshold=0.03, n_clusters=3)

# mean_model = MeanShift()
optics_model = OPTICS(eps=3, min_samples=4) #eps=0.75, min_samples=10
dbscan_model = DBSCAN(eps=3, min_samples=4) #0.25 9 eps=3.5, min_samples=4
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=4, gen_min_span_tree=True)


def kmeans_test(X,n):
    kmeans_model = KMeans(n_clusters=n)
    # kmeans_model.fit(X)
    # kmeans_result = kmeans_model.predict(X)
    # kmeans_clusters = unique(kmeans_result)
    # kmeans_result = bool_(kmeans_result)
    predict = kmeans_model.fit_predict(X)
    # labels = kmeans_model.labels_
    unique_labels = set(predict)
    # print("kmean_unique:",unique_labels)
    # centers = kmeans_model.cluster_centers_
    # print("kmeans centers:",centers)

    # plt.scatter(X[:, 0], X[:, 1], c=predict)
    # plt.plot(centers[:,0],centers[:,1],"r+",markersize=10.,markeredgewidth=5.)
    # plt.show()
    return predict,unique_labels
def dbscan_test(X):
    dbscan_result = dbscan_model.fit(X)
    # print(dbscan_result,type(X))
    labels = dbscan_result.labels_
    unique_labels = set(labels)
    return labels,unique_labels

def GMM_test(X,n):
    gaussian_model = GaussianMixture(n_components=n, covariance_type="full", max_iter=20, random_state=0)

    predict = gaussian_model.fit_predict(X)
    # print(predict,len(predict))
    unique_labels = set(predict)
    # print("unique_labels:",unique_labels)
    # plt.scatter(X[:, 0], X[:, 1], c=predict)
    # plt.show()
    return predict,unique_labels

def OPTICS(X):
    predict = optics_model.fit_predict(X)
    unique_labels = set(predict)
    # plt.scatter(X[:, 0], X[:, 1], c=predict)
    # plt.show()
    return predict, unique_labels

def Agglomerative(X):
    predict = agglomerative_model.fit_predict(X)
    unique_labels = set(predict)
    # print("Agglomerative:",unique_labels)
    # plt.scatter(X[:,0],X[:,1],c=predict)
    # plt.show()
    return predict,unique_labels

def HDBSCAN(X):
    predict = hdbscan_model.fit_predict(X)
    unique_labels = set(predict)
    # print("HDBSCAN:",unique_labels)
    # plt.scatter(X[:,0],X[:,1],c=predict)
    # plt.show()
    return predict,unique_labels

# if __name__=="__main__":
#     data_centers = [[1, 1], [-1, -1], [1, -1]]
#     dataset, labels_true = make_blobs(
#         n_samples=750, centers=data_centers, cluster_std=0.4, random_state=0
#     )
#     dataset_train = StandardScaler().fit_transform(dataset)
    # print("datatset_train:",dataset_train[0:30])
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
    # kmeans_test(dataset_train,kmeans_model)
    # dbscan_test(dataset_train)
    # GMM_test(dataset_train)
    # OPTICS(dataset_train)
    # Agglomerative(dataset_train)
    # HDBSCAN(dataset_train)