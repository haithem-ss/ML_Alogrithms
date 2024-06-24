import numpy as np


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def update_clusters(X, clusters,K):
    for i in range(K):
        points = np.array(clusters[i]["points"])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]["center"] = new_center
            clusters[i]["points"] = []
    return clusters


def assign_clusters(X, clusters,K):
    for i in range(X.shape[0]):
        dist = []
        curr_x = X[i]
        for j in range(K):
            dis = distance(curr_x, clusters[j]["center"])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]["points"].append(curr_x)
    return clusters


def pred_cluster(X, clusters,k):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]["center"]))
        pred.append(np.argmin(dist))
    return pred


def check_has_converged(old, new):
    for i in range(len(old)):
        if not np.array_equal(old[i], new[i]):
            return False
    return True
