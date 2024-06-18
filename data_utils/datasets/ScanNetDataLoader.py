from calendar import c
from re import T
import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from data_utils.data_map import *
from pcd_utils.pcd_preprocess import *
from pcd_utils.pcd_transforms import *
from pcd_utils.ScanNet_Choose_Segment_Fun import *
import random
import copy
from plyfile import PlyData
warnings.filterwarnings('ignore')

class ScanNetDataLoader(Dataset):
    def __init__(self, root, segment_pathbase, box_pathbase, split='train', args=None):
        # split指当前的状态，在Pretrain的状态中，我们不考虑使用Validation dataset, using only train split.所以
        self.args = args
        self.root = root
        self.segment_root = segment_pathbase
        self.box_root = box_pathbase
        
        # basic dataset infomation
        self.points_datapath = []
        self.split = split
        assert (split == 'train' or split == 'validation')
        self.datapath_list()
        self.exchange_radio=self.args["exchange_radio"]

        # loading cluster labels across the whole dataset
        print("total scene num is", len(self.points_datapath))


    def read_txt(self, path):
        """Read txt file into lines.
        """
        with open(path) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        return lines

    def num_to_natural(self, group_ids):
        '''
        Change the group number to natural number arrangement
        '''
        if np.all(group_ids == -1):
            return group_ids
        array = copy.deepcopy(group_ids).astype(np.int32)
        unique_values = np.unique(array[array != -1])
        mapping = np.full(int(np.max(unique_values) + 2), -1)
        mapping[unique_values + 1] = np.arange(len(unique_values))
        array = mapping[array + 1]
        return array


    def segments_box_path_update(self, params):
        self.segment_datapath = []
        self.box_path = []
        for i in range(len(self.train_file_path)):
            scan_path = self.train_file_path[i]
            scan_name = scan_path[:-4]
            self.segment_datapath += [ os.path.join(params["segment_save_path"], scan_name + '.npy') ]
            self.box_path += [ os.path.join(params["box_save_path"], scan_name + '_box.npy') ]     
        print("finsih update segement and box path, the base segment_path is ", params["segment_save_path"], "box basepath is ", params["box_save_path"])
        

    def datapath_list(self):
        self.points_datapath = []
        # self.labels_datapath = []
        self.segment_datapath = []
        self.near_relation = []
        self.box_path = []

        train_file_name = 'scannetv2_train.txt'
        train_file_path = self.read_txt(os.path.join(self.root, train_file_name))
        if self.args["debug_flag"]:
            train_file_path = train_file_path[:10]
            print("-------------------WE ARE DEBUGING--------------")

        self.train_file_path = train_file_path

        for i in range(len(train_file_path)):
            scan_path = train_file_path[i]
            scan_name = scan_path[:-4]
            
            mesh_file = os.path.join('scans_processed', 'train', scan_path)
            self.points_datapath += [ os.path.join(self.root, mesh_file)]
            self.segment_datapath += [ os.path.join(self.segment_root, scan_name + '.npy') ]

            self.box_path += [ os.path.join(self.box_root, scan_name + '_box.npy') ]


        print("finsih datapath obtain")



    def rotate_aug(self, points):

        points = np.expand_dims(points, axis=0)
        try:
            points[:,:,:3] = rotate_point_cloud(points[:,:,:3])  # 
        except:
            print("points:", points.shape)
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3]) # 
        return np.squeeze(points, axis=0)



    def transforms_cluster_wise(self, points):
        points = random_sample_cluster(points, strength=0.8)
        points = np.expand_dims(points, axis=0)
        return np.squeeze(points, axis=0)
    


    def load_ply(self, index):

        # filepath = self.data_root / self.SUBSET_PATH / self.data_paths[index]
        filepath = self.points_datapath[index]
        plydata = PlyData.read(filepath)
        data = plydata.elements[0].data
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
        labels = np.array(data['label'], dtype=np.int32)
        return coords, feats, labels, None




        

    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])  # 围绕Y轴的旋转 cluster-level的时候不做这个
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3]) # 三个方向上的小旋转
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])   # 尺度变换
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3]) # Y轴翻转
        points[:,:,:3] = jitter_point_cloud(points[:,:,:3])  # 随机抖动 点
        # points[:,:,3:6] = jitter_point_cloud(points[:,:,3:6])  # 随机抖动 颜色

        points = random_drop_n_cuboids(points)[0]   # 随机丢弃

        return np.squeeze(points, axis=0)




    def __len__(self):
        return len(self.points_datapath) 
        # return 10
    
    def _get_augmented_item(self, index):
        
            index_2 = np.random.choice(np.arange(0, len(self.points_datapath)), 1, replace=False)[0]
            
            while index_2 == index:
              
              index_2 = np.random.choice(np.arange(0, len(self.points_datapath)), 1, replace=False)[0]
              
            #   print("since index2 == index , resample one")
              
            
            # obtain data from scene 1
             
            coords, feats, labels, center = self.load_ply(index)

            feats[:, :3] = feats[:, :3] / 255. - 0.5 # normalize color just like STSeg
                
            segments_datapath = self.segment_datapath[index]

            points1 = np.concatenate([coords, feats], axis=1)
                                                  
            cluster_label_1 = np.load(segments_datapath).astype(int)                # 此时a是一个字典对象
                        
            box_1_path = self.box_path[index]
            
            box_1 = np.load(box_1_path)
          
            cluster_label_1 = np.array(cluster_label_1).reshape(-1, 1)
            
            cluster_label_1 = self.num_to_natural(cluster_label_1)
            
            points1 = np.concatenate([points1, cluster_label_1], axis=1)
            
            # obtain data from scene 2
            
            coords, feats, labels, center = self.load_ply(index_2)

            feats[:, :3] = feats[:, :3] / 255. - 0.5 # normalize color just like STSeg
                
            segments_datapath = self.segment_datapath[index_2]

            points2 = np.concatenate([coords, feats], axis=1)
            
            cluster_label_2 = np.load(segments_datapath).astype(int)                 # 此时a是一个字典对象
            
            box_2_path = self.box_path[index_2]
            
            box_2 = np.load(box_2_path)
          
            cluster_label_2 = np.array(cluster_label_2).reshape(-1, 1) 
            
            cluster_label_2 = self.num_to_natural(cluster_label_2) + np.max(cluster_label_1) + 1

            points2 = np.concatenate([points2, cluster_label_2], axis=1)          
           

            # segment id is unique index for each objects
            max_segment_index_1 = np.max(cluster_label_1)
            min_segment_index_2 = np.min(cluster_label_2)
            assert max_segment_index_1 < min_segment_index_2
            fliter_n1 = False
        
            
            points_save_1 = copy.deepcopy(points1)
            points_save_2 = copy.deepcopy(points2)


            if self.exchange_radio != 0:
                points1, points2 = exchange_object(points1, points2, points1[:, -1], points2[:, -1], box_1, box_2, model='size', min_points_limit=300, exchange_radio=self.exchange_radio, fliter_n1=fliter_n1)
                

            temp_zeros1 = np.zeros(shape=(points1.shape[0], 1))
            temp_zeros1[points1[:, -1] > max_segment_index_1, 0] = 1 # pin the objects not from this scene

            temp_zeros2 = np.zeros(shape=(points2.shape[0], 1))
            temp_zeros2[points2[:, -1] < min_segment_index_2, 0] = 1 
            temp_zeros2[points2[:, -1] == -1, 0] = 0 # -1视作自身部分
            # points_new = np.concatenate((points1, points2), axis=0)
            
            points_new_1 = np.concatenate((points1[:, 0:-1].copy(), temp_zeros1), axis=1)
            points_new_1 = np.concatenate((points_new_1, points1[:, -1].copy().reshape(-1, 1)), axis=1)

            points_new_2 = np.concatenate((points2[:, 0:-1].copy(), temp_zeros2), axis=1)
            points_new_2 = np.concatenate((points_new_2, points2[:, -1].copy().reshape(-1, 1)), axis=1)
    




            # # --------- for debuging ----------#
            # print("-------------------------------for debuging, no cuboid drop point clouds----------------------")
            # points_i = copy.deepcopy(points_new_1)      
            # points_i = self.transforms(points_i)   
            # points_j = self.transforms(points_save_1.copy())
            
            
            # points_i_2 = copy.deepcopy(points_new_2)      
            # points_i_2 = self.transforms(points_i_2)   
            # points_j_2 = self.transforms(points_save_2.copy())
                        
                        
            points_i = copy.deepcopy(points_new_1)      
            points_i = random_cuboid_point_cloud(points_i, s=0.9)[0]            # s=0.9？      
            points_i = self.transforms(points_i)   
            points_j = random_cuboid_point_cloud(points_save_1.copy())[0]                       
            points_j = self.transforms(points_j)
            
            
            points_i_2 = copy.deepcopy(points_new_2)      
            points_i_2 = random_cuboid_point_cloud(points_i_2, s=0.9)[0]                 
            points_i_2 = self.transforms(points_i_2)   
            points_j_2 = random_cuboid_point_cloud(points_save_2.copy())[0]                       
            points_j_2 = self.transforms(points_j_2)
            
            
                
            return points_i, points_j, points_i_2, points_j_2   # i是Student, j是Teacher



    def get_item(self, index):
        # points_datapath = self.points_datapath[index]
        self.index = index
        coords, feats, labels, center = self.load_ply(index)
        feats[:, :3] = feats[:, :3] / 255. - 0.5 # normalize color just like STSeg
        points = np.concatenate([coords, feats], axis=1)
        
        
        segments_datapath = self.segment_datapath[index]
        cluster_label_1 = np.load(segments_datapath).astype(int)  # 此时a是一个字典对象
        points = np.concatenate([points, cluster_label_1.reshape(-1, 1)], axis=1)
        
        points = random_cuboid_point_cloud(points, s=0.5)[0]                 
        points = self.transforms(points)   
        
        return points




    def __getitem__(self, index):
      
        points_i, points_j, points_i_2, points_j_2 = self._get_augmented_item(index)
        return points_i, points_j, points_i_2, points_j_2





