import open3d as o3d
import numpy as np
import cv2


def generate_box(points, segments, save_path):
    
    
    info_list_all = []
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(points[:, 0:3])
    
    cluster_label_list = np.unique(segments)

    # 处理第一个
    cluster_index_sample_index = np.where(segments == cluster_label_list[0])
    points_cluster = points[cluster_index_sample_index]
    temp_points = np.zeros(shape=(points_cluster.shape[0], 7))
    points_with_box = np.concatenate((points_cluster, temp_points), axis=1)


    for i in range(0, cluster_label_list.shape[0]):

        sample_index = cluster_label_list[i]

        cluster_index_sample_index = np.where(segments == sample_index)

        points_cluster = points[cluster_index_sample_index]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_cluster[:, 0:3])
        aabb = pcd.get_axis_aligned_bounding_box()  # get_axis_aligned_bounding_box  get_oriented_bounding_box
        try:
            rect = cv2.minAreaRect(np.float32(points_cluster[:, 0:2]))
        except:
            rect = cv2.minAreaRect(np.int32(points_cluster[:, 0:2]))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # box_center_3d = np.array([rect[0][0], rect[0][1], aabb.center[2]])
        box_center_3d = np.float64(np.array([rect[0][0], rect[0][1], (aabb.max_bound[2] + aabb.min_bound[2])/2])).reshape(-1, 1)
        angle=[0,0,0]
        # angle = rotationMatrixToAngles(aabb_oriented.R)
        angle[0] = 0
        angle[1] = 0
        angle[2] = rect[2]
        angle = np.float64(np.array(angle))
        R = aabb.get_rotation_matrix_from_xyz(angle*np.pi/180)
        aabb_oriented = o3d.geometry.OrientedBoundingBox(center=box_center_3d, R=R, extent=np.float64(np.array([rect[1][0], rect[1][1], (abs(aabb.max_bound[2]) - abs(aabb.min_bound[2]))])))
        info_list = [aabb_oriented.center[0],aabb_oriented.center[1],aabb_oriented.center[2], aabb_oriented.extent[0], aabb_oriented.extent[1], aabb_oriented.extent[2], rect[2]]
       
        box_info = np.array(info_list)
        box_info = box_info.reshape(1,-1).repeat(len(cluster_index_sample_index[0]),axis=0)

        temp_points = np.concatenate((points_cluster, box_info), axis=1)
        points_with_box = np.concatenate((points_with_box, temp_points), axis=0) # 将维度先扩展出来
        
        info_list_all.append(info_list)
        
    np.save(save_path, np.array(info_list_all))
    # print(save_path, out)
        # ScanNetdataset.points_datapath[j].rfind('/')


