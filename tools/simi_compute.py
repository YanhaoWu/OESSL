import os
from re import T 
import numpy as np
# from scipy import spatial

num = 10

feature = np.load('/home/WuYanhao/WorkSpace/EOSSL_PLUS/workdir/101/0/all_features.npy', allow_pickle=True)
simi_mat = []
for i in range(feature.shape[0]):
    simi_mat.append([])
    for j in range(feature.shape[0]):
        simi_mat[i].append([])
        cos_sim_sum = 0
        for k in range(num):
            random_select_1 = np.random.randint(len(feature[i]))
            random_select_2 = np.random.randint(len(feature[j]))
            vec1 = feature[i][random_select_1]
            vec2 = feature[j][random_select_2]
            cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_sim_sum = cos_sim_sum + cos_sim
        simi_mat[i][j] = cos_sim_sum / num


np.save("/home/WuYanhao/WorkSpace/EOSSL_PLUS/workdir/101/0/simimat_ob.npy", np.array(simi_mat))
print("Finish computing")