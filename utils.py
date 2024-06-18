import numpy as np
from data_utils.collations import *
from data_utils.datasets.ScanNetDataLoader import ScanNetDataLoader
from models.minkunet import *
from models.oessl import *
from models.blocks import ProjectionHead, SegmentationClassifierHead, PredictionHead, Exchange_predction_Layer
from data_utils.data_map import content, content_indoor
import os 


data_loaders = {
    'ScanNet': ScanNetDataLoader,
}

data_class = {
    'SemanticKITTI': 20,
    'SemanticPOSS': 14,
}

def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

def list_parameters(models):
    optim_params = []
    for model in models:
        optim_params += list(models[model].parameters())

    return optim_params


def get_projection_head(args, dtype):
    return ProjectionHead(in_channels=latent_features[args.sparse_model], out_channels=args.feature_size).type(dtype)


def get_oessl_model(args):
    return oessl_model(MinkUNet, ProjectionHead, ProjectionHead, PredictionHead, Exchange_predction_Layer, args)  # project 和 prodict 是一样的


def get_classifier_head(args, dtype):
    return SegmentationClassifierHead(
            in_channels=latent_features[args.sparse_model], out_channels=data_class[args.dataset_name]
        ).type(dtype)

def get_optimizer(optim_params, args):
    if 'UNet' in args.sparse_model:
        optimizer = torch.optim.SGD(optim_params, lr=args.lr, momentum=0.9, weight_decay=args.decay_lr)
    else:
        optimizer = torch.optim.Adam(optim_params, lr=args.lr, weight_decay=args.decay_lr)

    return optimizer

def get_class_weights(dataset):
    weights = list(content.values()) if dataset == 'SemanticKITTI' else list(content_indoor.values())

    weights = torch.from_numpy(np.asarray(weights)).float()
    if torch.cuda.is_available():
        weights = weights.cuda()

    return weights

def write_summary(writer, summary_id, report, epoch):
    writer.add_scalar(summary_id, report, epoch)

def get_dataset(args, segment_path, box_path, split='train'):
    data_train = ScanNetDataLoader(root=args["data_dir"], segment_pathbase=segment_path, box_pathbase=box_path, split=split, args=args)
    return data_train

def get_data_loader(dataset, batch_size, voxel_size, num_points, shuffle, num_workers, train_val):

    collate_fn = SparseAugmentedExhangeCollation(voxel_size, num_points) # 

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers
        )


    return data_loader





def make_dir(cfg, stage, save_index=None):
  
    # Set rounds 
    if save_index is not None:
        cfg["feature_save_base_path"] = os.path.join(cfg["feature_save_base_path"], str(save_index))
        cfg["segment_save_base_path"] = os.path.join(cfg["segment_save_base_path"], str(save_index))
        cfg["box_save_base_path"] = os.path.join(cfg["box_save_base_path"], str(save_index))
        cfg["object_feature_save_base_path"] = os.path.join(cfg["object_feature_save_base_path"], str(save_index))
        cfg["save_base_dir"] = os.path.join(cfg["save_base_dir"], str(save_index)) + '/'   
        cfg["log_dir"] = os.path.join(cfg["save_base_dir"], 'log') + '/'
        cfg["cluster_info_save_base_path"] = os.path.join(cfg["cluster_info_save_base_path"], str(save_index))


    # for stage_part
    cfg["feature_save_path"] = os.path.join(cfg["feature_save_base_path"], str(stage))
    cfg["segment_save_path"] = os.path.join(cfg["segment_save_base_path"], str(stage))
    cfg["box_save_path"] = os.path.join(cfg["box_save_base_path"], str(stage))
    cfg["object_feature_save_path"] = os.path.join(cfg["object_feature_save_base_path"], str(stage))
    cfg["save_dir"] = os.path.join(cfg["save_base_dir"], str(stage)) + '/'
    cfg["cluster_path"] = os.path.join(cfg["cluster_info_save_base_path"], str(stage))

    # making folders
    if not os.path.isdir(cfg["segment_save_path"]):
        os.makedirs(cfg["segment_save_path"])
    if not os.path.isdir(cfg["feature_save_path"]):
        os.makedirs(cfg["feature_save_path"])
    if not os.path.isdir(cfg["box_save_path"]):
        os.makedirs(cfg["box_save_path"])                
    if not os.path.isdir(cfg["object_feature_save_path"]):
        os.makedirs(cfg["object_feature_save_path"])            
    if not os.path.isdir(cfg["save_dir"]):
        os.makedirs(cfg["save_dir"])
    if not os.path.isdir(cfg["cluster_path"]):
        os.makedirs(cfg["cluster_path"])   
    
    
    return  cfg