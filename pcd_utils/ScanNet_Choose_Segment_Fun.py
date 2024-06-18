# 2022.7.14
# V1版
# 观察segcontrast的feature是否有一些有意义的东西

import numpy as np
from plyfile import PlyData
from pcd_utils.pcd_preprocess import *

from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans




def fileter_box(box, min=0.1, max=2.5, fliter_n1=False):  # 只要满足一定大小关系的Box  本来是0.3 ， 2.5
    box_index = []
    box_sample = []
    for i in range(box.shape[0]):
      if fliter_n1:
        if i != 0:      # 滤掉-1的时候，第一个box的segment应该就是-1，不要了就行
          box_size = box[i][3:6]
          if box_size[0] > min and box_size[1] > min and box_size[2] > min:
              if box_size[0] < max and box_size[1] < max and box_size[2] < max:
                  box_index.append(i)
                  box_sample.append(box[i])
      else:
          box_size = box[i][3:6]
          if box_size[0] > min and box_size[1] > min and box_size[2] > min:
              if box_size[0] < max and box_size[1] < max and box_size[2] < max:
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
        p0 = np.random.randint(0, n - 1, 1)
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






def Choose_Segment(center_info, unique_segment, method='both',exchange_radio=0.5):


    sample_flag = method
    if sample_flag == 'FPS':
        select_p, _ = FPS(center_info, int(len(unique_segment) * exchange_radio))
        select_p = unique_segment[select_p]
    elif sample_flag == 'Random':
        select_p = np.random.choice(unique_segment, int(len(unique_segment) * exchange_radio))
    elif sample_flag.lower() == 'both':
        select_p_1, _ = FPS(center_info, int(len(unique_segment) * (exchange_radio/2)))
        select_p_1 = unique_segment[select_p_1].tolist()  # FPS 因为是绝对下标，所以downsample的时候需要进行一次映射
        unique_segment = unique_segment.tolist()
        for k in range(len(select_p_1)):
            unique_segment.remove(select_p_1[k])
        select_p_2 = np.random.choice(unique_segment, int(len(unique_segment) * (exchange_radio/2)), replace=False)
        select_p_1.extend(select_p_2)
        select_p = select_p_1
    return select_p





def center_compute_one_segment(points):
    x_ave = np.average(points[:, 0])
    y_ave = np.average(points[:, 1])
    z_ave = np.average(points[:, 2])
    location_center = [x_ave, y_ave, z_ave]
    return np.array(location_center)








def center_compute(points, segment):
    center_info = []
    unique_segment = np.unique(segment)
    for j in range(len(unique_segment)):
        segment_j = unique_segment[j]
        segment_j_index = (segment == segment_j)
        x_ave = np.average(points[segment_j_index, 0])
        y_ave = np.average(points[segment_j_index, 1])
        z_ave = np.average(points[segment_j_index, 2])
        location_center = [x_ave, y_ave, z_ave]
        center_info.append(location_center)
    center_info = np.array(center_info)
    return center_info


def Compute_XYZ(points1, points2, segment1, segment2):
    center_info1 = center_compute(points1, segment1)
    center_info2 = center_compute(points2, segment2)

    return center_info1, center_info2


def exchange_object(points1, points2, segment1, segment2, box_1=None, box_2=None, model='size', colors1=None, colors2=None, min_points_limit=300, exchange_radio=0.5, fliter_n1 = False):
    center_info1, center_info2 = Compute_XYZ(points1, points2, segment1, segment2)

    if model == 'z':
        choose_index = Choose_Segment(center_info1, segment1)
        scene1_feature = center_info1[:, -1].reshape(-1, 1)
        scene2_feature = center_info2[:, -1].reshape(-1, 1)
        feature_dist = cdist(scene1_feature, scene2_feature)
        row_sort_index = np.argsort(feature_dist, axis=1)  # 行排序


    elif model == 'size':
        # scene1_feature = center_info1[:, -1].reshape(-1, 1)
        # scene2_feature = center_info2[:, -1].reshape(-1, 1)

        box_1_index, box_1_sample = fileter_box(box_1, fliter_n1=fliter_n1)
        if len(box_1_index) < 2:
            box_1_index, box_1_sample = fileter_box(box_1, min=0.2, max=3, fliter_n1=fliter_n1)
            # print("try resamping")
            if len(box_1_index) < 2:
                box_1_sample = box_1
                box_1_index = np.arange(0, box_1.shape[0]).tolist()
                # print("failed sampeing, use all boxs")

        box_2_index, box_2_sample = fileter_box(box_2, fliter_n1=fliter_n1)
        if len(box_2_index) < 2:
            box_2_index, box_2_sample = fileter_box(box_2, min=0.2, max=3, fliter_n1=fliter_n1)
            # print("try resamping")
            if len(box_2_index) < 2:
                box_2_sample = box_2
                box_2_index = np.arange(0, box_2.shape[0]).tolist()
                # print("failed sampeing, use all boxs")


        center_info1_sample = center_info1[box_1_index]
        center_info2_sample = center_info2[box_2_index]
        unique_segment1 = np.array(box_1_index)
        
        # ablation study part, test the useage of exchange radio
                
        choose_index = Choose_Segment(center_info1_sample, unique_segment1.copy(), method='both', exchange_radio=exchange_radio)

        
        try:
          sorted_index_row, sorted_index_colum, sorted_dist_mat_row, sorted_dist_mat_colum = box_dist(
              box_1_sample[:, 3:6].copy(), box_2_sample[:, 3:6].copy())
        except:
          print("box_1_sample", box_1_sample)
          print("box_2_sample", box_2_sample)
          
    if colors1 is not None:
        points1, points2, colors1, colors2 = exchange_segment(points1, points2, segment1, segment2, box_1_index, box_2_index, choose_index, sorted_index_row, colors1, colors2) # 用行排序的结果就可以
        return points1, points2, colors1, colors2

    else:
        points1, points2 = exchange_segment(points1, points2, segment1, segment2, box_1_index, box_2_index, choose_index, sorted_index_row, min_points_limit=min_points_limit) # 用行排序的结果就可以

        return points1, points2


def exchange_segment(points1, points2, segment1, segment2, box_1_index, box_2_index, choose_index, row_sort_index, colors1=None, colors2=None, z_axis_keep=False, min_points_limit=300):
    points_save1 = copy.deepcopy(points1)
    points_save2 = copy.deepcopy(points2)

    segment_new_1 = segment1
    segment_new_2 = segment2

    exchange_seg1_points_all = []
    exchange_seg2_points_all = []
    if colors1 is not None:
        exchange_seg1_colors_all = []
        exchange_seg2_colors_all = []

    pcd2_index_choose_all = []
    min_segment2 = int(np.min(segment2[segment2 != -1]))

    # for i in range(0, len(choose_index)):

    for i in range(0, min(20, len(choose_index))):

        # 挑选用于交换的Segment
        exchange_seg1_points = points1[segment_new_1 == choose_index[i]]
        if colors1 is not None:
            exchange_seg1_colors = colors1[segment_new_1 == choose_index[i]]

        exchange_seg1_points_z_ave = np.average(exchange_seg1_points, axis=0).reshape(1, -1)[0, -1]

        if exchange_seg1_points.shape[0] > min_points_limit and exchange_seg1_points_z_ave > 0.2:  

            for j in range(0, row_sort_index.shape[1]):
                choose_index_i = np.array(choose_index).astype(np.int)[i]       # 首选判断是选择的原始Segment中的哪一个
                sample_index_i = box_1_index.index(choose_index_i)              # 随后判断这个值在降采样之后的第几位，也就是对应的row_sort_index里的下标了
                corr_index_i = row_sort_index[sample_index_i][j]                # 对应的需要交换的在row_sort_index里的下标，下一步对应到原始的下标中的去
                pcd2_index_choose = box_2_index[corr_index_i] + min_segment2    # 和index这样才能对上
                exchange_seg2_points = points2[segment_new_2 == pcd2_index_choose]
                if colors1 is not None:
                    exchange_seg2_colors = colors2[segment_new_2 == pcd2_index_choose]

                if exchange_seg2_points.shape[0] > 0:
                    exchange_seg2_points_z_ave = np.average(exchange_seg2_points, axis=0).reshape(1, -1)[0, -1]
                else:
                    exchange_seg2_points_z_ave = -50
                if (pcd2_index_choose not in pcd2_index_choose_all) and (exchange_seg2_points.shape[0] > 300) and (
                        exchange_seg2_points_z_ave > 0.2):
                    pcd2_index_choose_all.append(pcd2_index_choose)

                    exchange_seg1_points_center = np.average(exchange_seg1_points, axis=0).reshape(1, -1)
                    exchange_seg2_points_center = np.average(exchange_seg2_points, axis=0).reshape(1, -1)

                    if z_axis_keep:

                        exchange_seg1_points[:, 0:2] = exchange_seg1_points[:, 0:2] - exchange_seg1_points_center[:,
                                                                                      0:2] + exchange_seg2_points_center[:,
                                                                                             0:2]
                        exchange_seg2_points[:, 0:2] = exchange_seg2_points[:, 0:2] - exchange_seg2_points_center[:,
                                                                                      0:2] + exchange_seg1_points_center[:,
                                                                                             0:2]
                    else:

                        exchange_seg1_points[:, 0:3] = exchange_seg1_points[:, 0:3] - exchange_seg1_points_center[:,
                                                                                      0:3] + exchange_seg2_points_center[:,
                                                                                             0:3]
                        exchange_seg2_points[:, 0:3] = exchange_seg2_points[:, 0:3] - exchange_seg2_points_center[:,
                                                                                      0:3] + exchange_seg1_points_center[:,
                                                                                             0:3]
                    exchange_seg1_points_all.extend(exchange_seg1_points.tolist())

                    exchange_seg2_points_all.extend(exchange_seg2_points.tolist())

                    if colors1 is not None:
                        exchange_seg1_colors_all.extend(exchange_seg1_colors.tolist())
                        exchange_seg2_colors_all.extend(exchange_seg2_colors.tolist())
                        colors1 = colors1[segment_new_1 != choose_index[i]]
                        colors2 = colors2[segment_new_2 != pcd2_index_choose]


                    # 对原始点云进行下采样
                    points1 = points1[segment_new_1 != choose_index[i]]
                    segment_new_1 = segment_new_1[segment_new_1 != choose_index[i]]

                    points2 = points2[segment_new_2 != pcd2_index_choose]
                    segment_new_2 = segment_new_2[segment_new_2 != pcd2_index_choose]



                    break
    exchange_seg2_points_all = np.array(exchange_seg2_points_all)
    exchange_seg1_points_all = np.array(exchange_seg1_points_all)

    try:    # some time there are two few objects 
        points1 = np.concatenate((points1, exchange_seg2_points_all), axis=0)
        points2 = np.concatenate((points2, exchange_seg1_points_all), axis=0)
        if colors1 is not None:
            colors1 = np.concatenate((colors1, exchange_seg2_colors_all), axis=0)
            colors2 = np.concatenate((colors2, exchange_seg1_colors_all), axis=0)
            return points1, points2, colors1, colors2
        else:
            return points1, points2

    except:
        
        # print("failed exchange, exchange_seg2_points_all is", exchange_seg2_points_all)
        # print("failed exchange, exchange_seg1_points_all is", exchange_seg1_points_all)
        # print("min points limit is", min_points_limit)

        
        
      
      
        if colors1 is not None:
            colors1 = np.concatenate((colors1, exchange_seg2_colors_all))
            colors2 = np.concatenate((colors2, exchange_seg1_colors_all))
            return points_save1, points_save2, colors1, colors2
        else:
            return points_save1, points_save2





def exchange_objeces_with_features(features1, segment1, features2, segment2):
  
    unique_segment_1 = np.unique(segment1)

    unique_segment_2 = np.unique(segment2)
    
    pooling_features_1 = []

    for i in range(0, len(unique_segment_1)):
      
      segment_i = unique_segment_1[i]

      segment_i_features = features1[ segment1 == segment_i ]
      
      segment_i_pooling_features = np.max(segment_i_features)

      pooling_features_1.append(segment_i_pooling_features)
      
    
    pooling_features_2 = []
    
    for j in range(0, len(unique_segment_2)):
      
      segment_j = unique_segment_2[j]

      segment_j_features = features1[ segment2 == segment_j ]
      
      segment_j_pooling_features = np.max(segment_j_features)

      pooling_features_2.append(segment_j_pooling_features)
    
    
    pooling_features_1 = np.asarray(pooling_features_1)
    
    pooling_features_2 = np.asarray(pooling_features_2)
    
    features_all = np.concatenate((pooling_features_1, pooling_features_2))

    estimator = KMeans(n_clusters=5, max_iter=200, n_init=10).fit(features_all)  # 构造聚类
    
    KM_Labels = estimator.labels_
    
    
 
def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array
 
 
def load_ply(points_datapath, index):

    # filepath = self.data_root / self.SUBSET_PATH / self.data_paths[index]
    filepath = points_datapath[index]
    plydata = PlyData.read(filepath)
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    labels = np.array(data['label'], dtype=np.int32)
    return coords, feats, labels, None 
 
 
 
 
 
 
    
if __name__ == "__main__":
  feauter_path1 = '/data/features/40/132/scene0000_00.npy'
  feauter_path2 = '/data/features/40/132/scene0005_00.npy'
  segment_path1 = '/data/Scans_Segment/40/132/scene0000_00.npy'
  segment_path2 = '/data/Scans_Segment/40/132/scene0005_00.npy'
  features1 = np.load(feauter_path1)
  segment1 = np.load(segment_path1)
  features2 = np.load(feauter_path2)
  segment2 = np.load(segment_path2)
  exchange_objeces_with_features(features1, segment1, features2, segment2)
  
  
  
  
  
  
  
  
  
  
  