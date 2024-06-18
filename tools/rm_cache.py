
import os
import yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"  

import argparse
import time
from numpy import inf




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--py_config', default='/home/WuYanhao/WorkSpace/EOSSL_PLUS/configs/cfg/scannet.yaml')
    parser.add_argument('--segment_path', default='/home/ssd/scans/Scans_Segment/large')
    parser.add_argument('--box_path', default='/home/ssd/scans/Scans_Box/box_large')
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--resume_path', default=None)
    parser.add_argument('--save_index', type=int, default=2, help='这是第几次进行训练')

    args = parser.parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    print(timestamp)
    with open(args.py_config, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    
    
    debug_flag = True 
    if debug_flag:
        cfg["update_epochs"].append(1)

    save_index = 1
    cfg["feature_save_path"] = os.path.join(cfg["feature_save_base_path"], str(save_index))
    cfg["segment_save_path"] = os.path.join(cfg["segment_save_base_path"], str(save_index))
    cfg["box_save_path"] = os.path.join(cfg["box_save_base_path"], str(save_index))
    cfg["object_feature_save_path"] = os.path.join(cfg["object_feature_save_base_path"], str(save_index))
    cfg["save_base_dir"] = os.path.join(cfg["save_base_dir"], str(save_index)) + '/'   
    cfg["cluster_path"] = os.path.join(cfg["cluster_info_save_base_path"], str(save_index))

    stage = 0
    # for stage_part
    cfg["feature_save_path"] = os.path.join(cfg["feature_save_path"], str(stage))
    cfg["segment_save_path"] = os.path.join(cfg["segment_save_path"], str(stage))
    cfg["box_save_path"] = os.path.join(cfg["box_save_path"], str(stage))
    cfg["object_feature_save_path"] = os.path.join(cfg["object_feature_save_path"], str(stage))
    cfg["save_dir"] = os.path.join(cfg["save_base_dir"], str(stage)) + '/'
    cfg["cluster_path"] = os.path.join(cfg["cluster_path"], str(stage))
    cfg["log_dir"] = os.path.join(cfg["save_dir"], 'log') + '/'

    print("Rrmoving ", cfg["save_dir"])
    command = 'rm -rf '
    os.system(command+cfg["segment_save_path"])
    os.system(command+cfg["feature_save_path"])
    os.system(command+cfg["box_save_path"])
    os.system(command+cfg["object_feature_save_path"])
    os.system(command+cfg["save_dir"])
    os.system(command+cfg["cluster_path"])


