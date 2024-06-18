import numpy as np 
import os
from tqdm import tqdm
from sklearn.cluster import KMeans

def generate_id(feature, segment, save_path, scene_id, object_id, index): 

    return scene_id, object_id
    print("FINISH")



def collect_object_features_with_semantic(feature, segment, label, all_object_feature): 
       
    assert feature.shape[0] == segment.shape[0]
    
    unique_segment = np.unique(segment)
    
    for j in range(len(unique_segment)):
                           
        feature_average = np.average(feature[(segment==unique_segment[j]).flatten()], axis=0)
        
        semantic_class = label[(segment==unique_segment[j]).flatten()].flatten()[0]
        
        if semantic_class != 255:
            
            all_object_feature[int(semantic_class)].append(feature_average)
    
    return all_object_feature
        



    
def generate_object_features(feature, segment, save_base_path): 
       
    assert feature.shape[0] == segment.shape[0]
    
    unique_segment = np.unique(segment)
    
    for j in range(len(unique_segment)):
        
        save_path = save_base_path + '_' + str(j) + '.npy'
                    
        feature_average = np.average(feature[(segment==unique_segment[j]).flatten()], axis=0)
        
        np.save(save_path, feature_average)
        
        
        

def cluster_features(feature_path, save_path):
    
    # 这个feature_path是object features path     
    
    object_list = os.listdir(feature_path)

    feature_path_all = [os.path.join(feature_path, feature[:-4]+'.npy') for feature in object_list]
    
    feature_all = []

    for index in tqdm(range(len(feature_path_all))):

        feature_all.append(np.load(feature_path_all[index]))
        
    feature_all = np.nan_to_num(np.array(feature_all))
    
    print("START K_MEANS CLUSTERING")

    kmeans = KMeans(n_clusters=50)
    
    kmeans.fit(feature_all)
    
    labels = kmeans.predict(feature_all)
    
    np.save(save_path, labels)
    
    print("FINISH COLLECTING FEATURES")
    
    
