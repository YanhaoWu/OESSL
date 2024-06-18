import numpy as np
import open3d as o3d
import copy
from scipy.spatial.distance import cdist






def fileter_box(box): # 只要满足一定大小关系的Box
    box_index = []
    box_sample = []
    for i in range(box.shape[0]):
        box_size = box[i][3:6]
        if box_size[0] > 0.5 and box_size[1] > 0.5 and box_size[2] > 0.5:
            box_index.append(i)
            box_sample.append(box[i])
    return box_index, np.array(box_sample)

def box_dist(box1, box2):
    dist_1 = cdist(box1, box2)
    temp_box_1 = box1.copy()
    temp_box_2 = box2.copy()
    temp_box_1[:, 0] = box1[:, 1]
    temp_box_1[:, 1] = box1[:, 0]
    dist_2 = cdist(temp_box_1, temp_box_2)
    dist_1_extend = np.expand_dims(dist_1, 0)
    dist_2_extend = np.expand_dims(dist_2, 0)
    dist_cat = np.concatenate((dist_1_extend, dist_2_extend), axis=0)

    min_dist = np.min(dist_cat, axis=0)
    sorted_index_row = np.argsort(min_dist, axis=1)  # 1 -> 2
    sorted_dist_mat_row = np.sort(min_dist, axis=1)  # 1 -> 2
    sorted_index_colum = np.argsort(sorted_dist_mat_row[:, 0], axis=0)
    sorted_dist_mat_colum = sorted_dist_mat_row[sorted_index_colum]
    sorted_index_row = sorted_index_row[sorted_index_colum]
    return sorted_index_row, sorted_index_colum, sorted_dist_mat_row, sorted_dist_mat_colum











def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def FPS(sample, num, random_init=True):
    '''sample:采样点云数据,
    num:需要采样的数据点个数'''
    n = sample.shape[0]
    center = np.mean(sample, axis=0)  # 点云重心
    select_p = []  # 储存采集点索引
    L = []
    for i in range(n):
        L.append(distance(sample[i], center))
    p0 = np.argmax(L)
    if random_init:
        p0 = np.random.randint(0, n-1, 1)
    else:
        p0 = np.argmax(L)
        select_p.append(p0)  # 选距离重心最远点p0
    L = []
    for i in range(n):
        L.append(distance(p0, sample[i]))
    select_p.append(np.argmax(L))
    for i in range(num - 2):
        for p in range(n):
            d = distance(sample[select_p[-1]], sample[p])
            if d <= L[p]:
                L[p] = d
        select_p.append(np.argmax(L))
    return select_p, sample[select_p]










def overlap_clusters(cluster_i, cluster_j, min_cluster_point=20):
    # get unique labels from pcd_i and pcd_j
    unique_i = np.unique(cluster_i)
    unique_j = np.unique(cluster_j)

    # get labels present on both pcd (intersection)
    unique_ij = np.intersect1d(unique_i, unique_j)[1:]

    # also remove clusters with few points
    for cluster in unique_ij.copy():
        ind_i = np.where(cluster_i == cluster)
        ind_j = np.where(cluster_j == cluster)

        if len(ind_i[0]) < min_cluster_point or len(ind_j[0]) < min_cluster_point:
            unique_ij = np.delete(unique_ij, unique_ij == cluster)
        
    # labels not intersecting both pcd are assigned as -1 (unlabeled)
    cluster_i[np.in1d(cluster_i, unique_ij, invert=True)] = -1
    cluster_j[np.in1d(cluster_j, unique_ij, invert=True)] = -1

    return cluster_i, cluster_j

def clusters_hdbscan(points_set, n_clusters):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=None
                            )

    clusterer.fit(points_set)

    labels = clusterer.labels_.copy()

    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:n_clusters, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusters_from_pcd(pcd, n_clusters):
    # clusterize pcd points
    labels = np.array(pcd.cluster_dbscan(eps=0.25, min_points=10))
    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:n_clusters, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusterize_pcd(points, n_clusters):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # segment plane (ground)
    _, inliers = pcd.segment_plane(distance_threshold=0.25, ransac_n=3, num_iterations=200)
    pcd_ = pcd.select_by_index(inliers, invert=True)

    labels_ = np.expand_dims(clusters_from_pcd(pcd_, n_clusters), axis=-1)

    # that is a blessing of array handling
    # pcd are an ordered list of points
    # in a list [a, b, c, d, e] if we get the ordered indices [1, 3]
    # we will get [b, d], however if we get ~[1, 3] we will get the opposite indices
    # still ordered, i.e., [a, c, e] which means listing the inliers indices and getting
    # the invert we will get the outliers ordered indices (a sort of indirect indices mapping)
    labels = np.ones((points.shape[0], 1)) * -1
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[inliers] = False

    labels[mask] = labels_

    retu\