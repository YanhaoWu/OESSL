
import os
import yaml
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

from trainer.oessl_trainer import oessl_trainer
from pytorch_lightning import Trainer
from utils import *
import argparse
import time
from numpy import inf




if __name__ == "__main__":

    print("Start OESSL training")

    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--py_config', default='configs/cfg/scannet.yaml')
    parser.add_argument('--segment_path', default='data/Scans_Segment/segments')
    parser.add_argument('--box_path', default='data/Scans_Box/box')
    parser.add_argument('--stage', type=int, default=0)dsa
    parser.add_argument('--resume_path', default=None)
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--save_index', type=int, default=0, help='这是第几次进行训练')
    parser.add_argument('--debug_flag', default=False)
    parser.add_argument('--test_flag', default=False)

    args = parser.parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    print(timestamp)
    with open(args.py_config, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    
    
    cfg['debug_flag'] = args.debug_flag
    cfg['load_path'] = args.load_path
    
    if args.debug_flag:
        cfg["update_epochs"].append(1)
        cfg["update_epochs"].append(2)
        cfg["update_epochs"].append(3)    
    
    cfg = make_dir(cfg, stage=args.stage, save_index=args.save_index)
    
    train_set = get_dataset(cfg, segment_path=args.segment_path, box_path=args.box_path, split='train')
    
    val_set = get_dataset(cfg, segment_path=args.segment_path, box_path=args.box_path, split='validation') 
    
    train_loader = get_data_loader(train_set,
                                    batch_size=cfg["training"]["batch_size"],
                                    voxel_size=cfg["training"]["voxel_size"],
                                    num_points=cfg["training"]["number_points"],
                                    shuffle=True,
                                    num_workers=cfg["threads"],
                                    train_val='train')
    
  
    criterion = nn.CrossEntropyLoss().cuda()

    model = get_oessl_model(cfg)  

    # save the run config
    print("load_path is ", args.resume_path)
    argsDict = args.__dict__
    config_save_path = cfg["save_dir"] +  'config.txt'
    print("saving config at", config_save_path)
    with open(config_save_path, 'w') as f:
      for eachArg, value in argsDict.items():
          f.writelines(eachArg + ' : ' + str(value) + '\n')
    yaml_save_path = cfg["save_dir"] + 'cfg.yaml'
    command = 'cp ' + args.py_config + ' ' + yaml_save_path
    os.system(command)
    
    
    model_lightning = oessl_trainer(model, criterion, train_loader, params=cfg)
    
    if args.resume_path is not None:
        trainer = Trainer(gpus=-1, accelerator='ddp', max_epochs=cfg["training"]["max_epochs"], accumulate_grad_batches=1, resume_from_checkpoint = args.resume_path, default_root_dir=cfg["save_dir"])
    else:
        trainer = Trainer(gpus=-1, accelerator='ddp', max_epochs=cfg["training"]["max_epochs"], accumulate_grad_batches=1, default_root_dir=cfg["save_dir"])

    if args.test_flag:
        trainer.test(model_lightning)
    else:
        trainer.fit(model_lightning)


