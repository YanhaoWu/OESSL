import numpy as np
import torch
from pcd_utils.pcd_preprocess import overlap_clusters
import MinkowskiEngine as ME

def array_to_sequence(batch_data):
        return [ row for row in batch_data ]

def array_to_torch_sequence(batch_data):
    return [ torch.from_numpy(row).float() for row in batch_data ]


def list_segments_number(segments):
    num_points = []
    for batch_num in range(segments.shape[0]):
        for segment_lbl in np.unique(segments[batch_num]):
            if segment_lbl == -1:
                continue
            segment_ind = np.where(segments[batch_num] == segment_lbl)[0]
            num_points.append(segment_ind.shape[0])
    return num_points

def list_segments_points(p_coord, p_feats, labels, collect_numbers=False):
    c_coord = []
    c_feats = []
    num_points = []
    seg_batch_count = 0

    for batch_num in range(labels.shape[0]):
        batch_ind = p_coord[:,0] == batch_num       # 
                                                    # 
        for segment_lbl in np.unique(labels[batch_num]):
            if segment_lbl == -1:
                continue
            # batch_ind = p_coord[:,0] == batch_num
            segment_ind = labels[batch_num] == segment_lbl

            # we are listing from sparse tensor, the first column is the batch index, which we drop
            segment_coord = p_coord[batch_ind][segment_ind][:,:]
            segment_coord[:,0] = seg_batch_count
            seg_batch_count += 1        # 每个聚类体单独看做一个Batch

            segment_feats = p_feats[batch_ind][segment_ind]

            c_coord.append(segment_coord)
            c_feats.append(segment_feats)
            num_points.append(segment_coord.shape[0])

    seg_coord = torch.vstack(c_coord)
    seg_feats = torch.vstack(c_feats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if collect_numbers:
        return ME.SparseTensor(
                    features=seg_feats,
                    coordinates=seg_coord,
                    device=device,
                ), num_points
    else:
        return ME.SparseTensor(
                    features=seg_feats,
                    coordinates=seg_coord,
                    device=device,
                )    




def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(array_to_torch_sequence(p_label), dtype=torch.float32)[:, 1:]
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label.cuda()

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )



def numpy_to_sparse_tensor_filed(p_coord, p_feats, p_label=None, sparse_resolution=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = array_to_sequence(p_coord)[0]
    p_feats = array_to_torch_sequence(p_feats)[0]

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(array_to_torch_sequence(p_label), dtype=torch.float32)[:, 1:]
  
        return  ME.TensorField(
                features=p_feats,
                coordinates=ME.utils.batched_coordinates([p_coord], dtype=torch.float32),
                device=device,
            ), p_label.cuda()

    return  ME.TensorField(
                features=p_feats,
                coordinates=ME.utils.batched_coordinates([p_coord], dtype=torch.float32),
                device=device,
            )





def point_set_to_coord_feats(point_set, labels, resolution, num_points, deterministic=False):
    p_feats = point_set.copy()[:, 3:]
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1)

    _, mapping = ME.utils.sparse_quantize(coordinates=np.ascontiguousarray(p_coord), return_index=True)
    if len(mapping) > num_points:
        if deterministic:
            # for reproducibility we set the seed
            np.random.seed(42)
        mapping = np.random.choice(mapping, num_points, replace=False)

    return p_coord[mapping], p_feats[mapping], labels[mapping]

def point_set_to_coord_feats_without_labels(point_set, resolution, num_points, deterministic=False, no_mapping=False):
    p_feats = point_set.copy()[:, 3:]
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1)
    
    p_coord = np.ascontiguousarray(p_coord)
    
    if no_mapping:
        return p_coord, p_feats

    else:    
        _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)
        if len(mapping) > num_points:
            if deterministic:
                # for reproducibility we set the seed
                np.random.seed(42)
            mapping = np.random.choice(mapping, num_points, replace=False)

        return p_coord[mapping], p_feats[mapping]




def collate_points_to_sparse_tensor(pi_coord, pi_feats, pj_coord, pj_feats):
    # voxelize on a sparse tensor
    points_i = numpy_to_sparse_tensor(pi_coord, pi_feats)
    points_j = numpy_to_sparse_tensor(pj_coord, pj_feats)

    return points_i, points_j

    




class SparseAugmentedExhangeCollation:
    def __init__(self, resolution, num_points=80000, segment_contrast=False):
        self.resolution = resolution
        self.num_points = num_points
        self.segment_contrast = segment_contrast


    def point_set_to_coord_feats_return_maapping(self, point_set, resolution, num_points, deterministic=False):
        p_feats = point_set.copy()[:, 3:]
        p_coord = np.round(point_set[:, :3] / resolution)
        p_coord -= p_coord.min(0, keepdims=1)
        mapping = None
        _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)
        mapping = np.asarray(mapping)
        if len(mapping) > num_points:
            if deterministic:
                # for reproducibility we set the seed
                np.random.seed(42)
            mapping = np.random.choice(mapping, num_points, replace=False)

        return p_coord[mapping], p_feats[mapping], mapping



    def deal_only_points(self, points_i_1, points_j_1, points_i_2, points_j_2):

        points_i_1 = np.asarray(points_i_1)
        points_j_1 = np.asarray(points_j_1)
        points_i_2 = np.asarray(points_i_2)
        points_j_2 = np.asarray(points_j_2)
        

        pi_feats_1 = []
        pi_exan_1 = []
        pi_coord_1 = []
        pi_map_1 = []

        pi_feats_2 = []
        pi_exan_2 = []
        pi_coord_2 = []
        pi_map_2 = []

        pj_feats_1 = []
        pj_exan_1 = []
        pj_coord_1 = []
        pj_map_1 = []
        
        
        pj_feats_2 = []
        pj_exan_2 = []
        pj_coord_2 = []
        pj_map_2 = []


        for index in range(0, len(points_i_1)):
          
            pi1 = points_i_1[index]
            pj1 = points_j_1[index]

            coord_pi_1, feats_pi_1, mapping_i_1 = self.point_set_to_coord_feats_return_maapping(pi1[:,:-1], self.resolution, self.num_points)
            pi_coord_1.append(coord_pi_1)
            
            pi_feats_1.append(feats_pi_1[:, :-1])
            pi_exan_1.append(feats_pi_1[:, -1].reshape(-1, 1))
            
            pi_map_1.append(mapping_i_1)

            coord_pj_1, feats_pj_1, mapping_j_1 = self.point_set_to_coord_feats_return_maapping(pj1[:,:-1], self.resolution, self.num_points)
            pj_coord_1.append(coord_pj_1)
            
            pj_feats_1.append(feats_pj_1)
            pj_map_1.append(mapping_j_1)

                        
            pi2 = points_i_2[index]
            pj2 = points_j_2[index]

            coord_pi_2, feats_pi_2, mapping_i_2 = self.point_set_to_coord_feats_return_maapping(pi2[:,:-1], self.resolution, self.num_points)
            pi_coord_2.append(coord_pi_2)
            
            pi_feats_2.append(feats_pi_2[:, :-1])
            pi_exan_2.append(feats_pi_2[:, -1].reshape(-1, 1))

            
            pi_map_2.append(mapping_i_2)



            coord_pj_2, feats_pj_2, mapping_j_2 = self.point_set_to_coord_feats_return_maapping(pj2[:,:-1], self.resolution, self.num_points)
            pj_coord_2.append(coord_pj_2)
            
            pj_feats_2.append(feats_pj_2)       

            pj_map_2.append(mapping_j_2)
            
                        
                        

        pi_feats_1 = np.asarray(pi_feats_1)
        pi_coord_1 = np.asarray(pi_coord_1)
        pi_map_1 = np.asarray(pi_map_1)
        pi_exan_1  = np.asarray(pi_exan_1)
        
        pj_feats_1 = np.asarray(pj_feats_1)
        pj_coord_1 = np.asarray(pj_coord_1)
        pj_map_1 = np.asarray(pj_map_1)

        

        pi_feats_2 = np.asarray(pi_feats_2)
        pi_coord_2 = np.asarray(pi_coord_2)
        pi_map_2 = np.asarray(pi_map_2)
        pi_exan_2  = np.asarray(pi_exan_2)


        pj_feats_2 = np.asarray(pj_feats_2)
        pj_coord_2 = np.asarray(pj_coord_2)
        pj_map_2 = np.asarray(pj_map_2)

        

        # if not segment_contrast segment_i and segment_j will be an empty list
        return (pi_coord_1, pi_feats_1, pi_map_1, pi_exan_1), (pj_coord_1, pj_feats_1, pj_map_1), (pi_coord_2, pi_feats_2, pi_map_2, pi_exan_2), (pj_coord_2, pj_feats_2, pj_map_2)





    def deal_only_segments(self, points_i_1, points_j_1, points_i_2, points_j_2, pi_map_1, pj_map_1, pi_map_2, pj_map_2):
        # 每个PCD应该两个Overlap的Segments
        s_i1_j1_i = []
        s_i1_j1_j = []
        
        s_i1_j2_i = []
        s_i1_j2_j = []
        
        s_i2_j2_i = []
        s_i2_j2_j = []
        
        s_i2_j1_i = []
        s_i2_j1_j = []
        
        for index in range(0, len(pi_map_1)):
          
          pi_map_1_index = points_i_1[index][:, -1][pi_map_1[index]]
          pj_map_1_index = points_j_1[index][:, -1][pj_map_1[index]]
          
          pi_map_2_index = points_i_2[index][:, -1][pi_map_2[index]]
          pj_map_2_index = points_j_2[index][:, -1][pj_map_2[index]]
          
          
          # i j
          
          segment_i1_j1_i, segment_i1_j1_j = overlap_clusters(pi_map_1_index.copy(), pj_map_1_index.copy()) # 保留部分
          s_i1_j1_i.append(segment_i1_j1_i)
          s_i1_j1_j.append(segment_i1_j1_j)
          
          segment_i1_j2_i, segment_i1_j2_j = overlap_clusters(pi_map_1_index.copy(), pj_map_2_index.copy()) # 交换部分
          s_i1_j2_i.append(segment_i1_j2_i)
          s_i1_j2_j.append(segment_i1_j2_j)

          
          segment_i2_j2_i, segment_i2_j2_j = overlap_clusters(pi_map_2_index.copy(), pj_map_2_index.copy()) # 保留部分
          s_i2_j2_i.append(segment_i2_j2_i)
          s_i2_j2_j.append(segment_i2_j2_j)

          
          segment_i2_j1_i, segment_i2_j1_j = overlap_clusters(pi_map_2_index.copy(), pj_map_1_index.copy()) # 交换部分
          s_i2_j1_i.append(segment_i2_j1_i)
          s_i2_j1_j.append(segment_i2_j1_j)
        
        s_i1_j1_i = np.asarray(s_i1_j1_i)
        s_i1_j1_j = np.asarray(s_i1_j1_j)
        
        s_i1_j2_i = np.asarray(s_i1_j2_i)
        s_i1_j2_j = np.asarray(s_i1_j2_j)

        s_i2_j2_i = np.asarray(s_i2_j2_i)
        s_i2_j2_j = np.asarray(s_i2_j2_j)
        
        s_i2_j1_i = np.asarray(s_i2_j1_i)
        s_i2_j1_j = np.asarray(s_i2_j1_j)
        
        return (s_i1_j1_i, s_i1_j1_j), (s_i1_j2_i, s_i1_j2_j), (s_i2_j2_i, s_i2_j2_j), (s_i2_j1_i, s_i2_j1_j)

          

    def __call__(self, list_data):

        points_i_1, points_j_1, points_i_2, points_j_2 = list(zip(*list_data))
       
        (pi_coord_1, pi_feats_1, pi_map_1, pi_exan_1), (pj_coord_1, pj_feats_1, pj_map_1), (pi_coord_2, pi_feats_2, pi_map_2, pi_exan_2), (pj_coord_2, pj_feats_2, pj_map_2) = self.deal_only_points(points_i_1, points_j_1, points_i_2, points_j_2)
        S_i1j1, S_i1j2, S_i2j2, S_i2j1 = self.deal_only_segments(points_i_1, points_j_1, points_i_2, points_j_2, pi_map_1, pj_map_1, pi_map_2, pj_map_2)
        
        return (pi_coord_1, pi_feats_1, pi_exan_1), (pj_coord_1, pj_feats_1), (pi_coord_2, pi_feats_2, pi_exan_2), (pj_coord_2, pj_feats_2),  (S_i1j1, S_i1j2, S_i2j2, S_i2j1)
              
