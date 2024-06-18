
import os
import numpy as np
import argparse
import subprocess
import sys
import time


parser = argparse.ArgumentParser(description='OESSLPLUS_Params')
parser.add_argument('--max_epoch', type=int, default=200, help='max epochs to run')
parser.add_argument('--save_index', type=int, default=1, help='where to save the pre-trained models')
parser.add_argument('--stop_time', type=int, default=3, help='how many times wo use features to help segmentation')
parser.add_argument('--exchange_radio', type=float, default=0.5, help='the exchange radio of the objects')
parser.add_argument('--balance_loss_radio', type=float, default=2, help='gama for auxilary task')
parser.add_argument('--gradual_radio', default=False, help='if we gradual increase radio')
parser.add_argument('--balance_remain_swap_radio', type=float, default=1, help='remain part and swap part')
args = parser.parse_args()


max_epoch = args.max_epoch
save_index = args.save_index
stop_time = args.stop_time
exchange_radio = args.exchange_radio
balance_loss_radio = args.balance_loss_radio
gradual_radio = args.gradual_radio
print(args)

segment_para1 = 1.5
segment_para2 = 300 # 1.5 / 300 large segments, 0.5 small segments
print("when using features to help segmenting, the params are ", segment_para1, segment_para2)
  
segment_save_path = '/data/Scans_Segment/large'
box_save_path = '/data/Scans_Box/box_large'

use_previous_stage = None # None / int(args)
save_index_cmd = '--save_index ' + str(save_index) + ' '
oldstdout = sys.stdout
model_save_dir = '/home/WuYanhao/WorkSpace/EOSSL_PLUS/workdir/' + str(save_index) 
if not(os.path.isdir(model_save_dir)):
    print("making dir", model_save_dir)
    os.makedirs(model_save_dir)
txt_path = model_save_dir + '/' + 'log.txt'
generate_path = model_save_dir + '/' + 'generate_log.txt'
segment_path = model_save_dir + '/' + 'seg_log.txt'
train_path = model_save_dir + '/' + 'train_path_log.txt'
box_path = model_save_dir + '/' + 'box.txt'

print("Start training")

file = open(txt_path ,'a')
file.flush()
generate_file = open(generate_path ,'a')
segment_file = open(segment_path ,'a')
train_file = open(train_path ,'a')
box_file = open(box_path ,'a')

sys.stdout = file
print("use_previous_stage ", use_previous_stage)
pre_stop_epoch = 0
# pre_stop_epoch = int(200 / stop_time * (0 + 1)) # 临时
for i in range(0, stop_time):
  print("i is ", i)
  stop_epoch = int(max_epoch / stop_time * (i + 1))
  # 第一次是直接执行
  if i != 0:
    load_epoch = str(int(max_epoch / stop_time * i) - 1)
    load_index = str(save_index)
    segment_save_path = '/home/WuYanhao/WorkSpace/data/Scans_Segment' + '/' + load_index + '/' + load_epoch
    box_save_path = '/home/WuYanhao/WorkSpace/data/Scans_Box' + '/' + load_index + '/' + load_epoch
    if gradual_radio:
      exchange_radio = np.floor(int(load_epoch) / 20) * 0.05 + 0.1
      print("exchange_radio is ", exchange_radio, file=oldstdout)
      
  cmd = 'python /home/WuYanhao/WorkSpace/EOSSL_PLUS/train.py '
  stage = '--stage ' + str(i)
  
  if abs(max_epoch - stop_epoch) < 10:
    stop_epoch = max_epoch
  if stop_epoch >= 200:
    stop_epoch = 200 # 保险
  stop_epoch = int(stop_epoch)
  stop_epoch_cmd = ' --stop_epoch ' + str(stop_epoch)
  pre_stop_epoch_cmd = ' --pre_stop_epoch ' + str(pre_stop_epoch)
  if gradual_radio:
    cmd = cmd + save_index_cmd + stage + stop_epoch_cmd + pre_stop_epoch_cmd + ' --box_path ' + box_save_path + ' --segment_path ' + segment_save_path + ' --exchange_radio ' + str(exchange_radio) + ' --balance_loss_radio' + str(balance_loss_radio) + ' --gradual_radio'
  else:
    cmd = cmd + save_index_cmd + stage + stop_epoch_cmd + pre_stop_epoch_cmd + ' --box_path ' + box_save_path + ' --segment_path ' + segment_save_path

  print(time.strftime('%Y-%m-%d %H:%M:%S')) #结构化输出当前的时间
  print(cmd)
  file.flush()

  print("starting pretraing")
  print("the segment_save_path is ", segment_save_path)
  print("the box_save_path is ", box_save_path)
  
  if gradual_radio:
    result = subprocess.run(['python', '/home/WuYanhao/WorkSpace/EOSSL_PLUS/train.py', '--stage', str(i), '--stop_epoch', str(int(stop_epoch)), '--save_index',  str(save_index), '--pre_stop_epoch', str(pre_stop_epoch), '--segment_path', segment_save_path, '--box_path', box_save_path, '--exchange_radio', str(exchange_radio), '--balance_loss_radio', str(balance_loss_radio), '--gradual_radio', '--balance_remain_swap_radio', str(args.balance_remain_swap_radio)],  stdout = train_file, check=True)
  else:
    result = subprocess.run(['python', '/home/WuYanhao/WorkSpace/EOSSL_PLUS/train.py', '--stage', str(i), '--stop_epoch', str(int(stop_epoch)), '--save_index',  str(save_index), '--pre_stop_epoch', str(pre_stop_epoch), '--segment_path', segment_save_path, '--box_path', box_save_path],  stdout = train_file, check=True)

  pre_stop_epoch = stop_epoch # 后面再更新
  print("finishing pretraing ", i)
  if i != (stop_time - 1):
    # generate features
    cmd_generate = 'python /home/wuyanhao/WorkSpace/STSSL_Features_Get/feature_points_get.py '
    load_from_index_cmd = '--load_from_index ' + str(save_index) + ' '
    best_epoch = stop_epoch - 1
    best_epoch_cmd = '--best epoch' + str(best_epoch) + ' '
    cmd_generate = cmd_generate + load_from_index_cmd + best_epoch_cmd
    print(time.strftime('%Y-%m-%d %H%M%S')) #结构化输出当前的时间
    print("generating features")
    feature_save_base_path = '/data/features' + '/' + str(save_index) + '/' + str(best_epoch)
    if not os.path.isdir(feature_save_base_path):
      os.makedirs(feature_save_base_path)
    
    result = subprocess.run(['python', '/home/wuyanhao/WorkSpace/STSSL_Features_Get/feature_points_get_for_train.py', '--best', ('epoch' + str(best_epoch)), '--load_from_index', str(save_index), '--features_save_path', feature_save_base_path], stdout = generate_file)

    # os.system(cmd_generate)


    file.flush()

    result_all = []
    # segment
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/EOSSL/tools/segmentator_python.py', '--start', '0', '--end', '200', '--load_index', str(save_index), '--load_epoch', str(best_epoch) ,'--para1', str(segment_para1), '--para2', str(segment_para2)], stdout = segment_file )
    result_all.append(result)
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/EOSSL/tools/segmentator_python.py', '--start', '200', '--end', '400', '--load_index', str(save_index), '--load_epoch', str(best_epoch) , '--para1', str(segment_para1), '--para2', str(segment_para2)], stdout = segment_file)
    result_all.append(result)
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/EOSSL/tools/segmentator_python.py', '--start', '400', '--end', '600', '--load_index', str(save_index), '--load_epoch', str(best_epoch) ,'--para1', str(segment_para1), '--para2', str(segment_para2)], stdout = segment_file)
    result_all.append(result)
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/EOSSL/tools/segmentator_python.py', '--start', '600', '--end', '800', '--load_index', str(save_index), '--load_epoch', str(best_epoch) ,'--para1', str(segment_para1), '--para2', str(segment_para2)], stdout = segment_file)
    result_all.append(result)
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/EOSSL/tools/segmentator_python.py', '--start', '800', '--end', '1000', '--load_index', str(save_index), '--load_epoch', str(best_epoch) ,'--para1', str(segment_para1), '--para2', str(segment_para2)], stdout = segment_file)
    result_all.append(result)
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/EOSSL/tools/segmentator_python.py', '--start', '1000', '--end', '1202', '--load_index', str(save_index), '--load_epoch', str(best_epoch) ,'--para1', str(segment_para1), '--para2', str(segment_para2)], stdout = segment_file)
    result_all.append(result)
    while(True):
      count = 0
      for i_count in range(len(result_all)):
        if result_all[i_count].poll() is None:
          count = 0 
          break;
        else:
          count+=1
      if count == len(result_all):
        break;
    print(time.strftime('%Y-%m-%d %H:%M:%S')) #结构化输出当前的时间
    print("Finish Segmenting")


    file.flush()


    load_epoch = str(best_epoch)
    load_index = str(save_index)
    segment_save_path = '/data/Scans_Segment' + '/' + load_index + '/' + load_epoch
    box_save_path = '/data/Scans_Box' + '/' + load_index + '/' + load_epoch
    if not os.path.isdir(box_save_path):
      os.makedirs(box_save_path)

    result_all = []
    # segment
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/STSSL/data_preprocessing/detection_box_generate_ScanNet.py', '--start', '0', '--end', '200', '--segment_save_path', segment_save_path, '--box_save_path', box_save_path], stdout = box_file )
    result_all.append(result)
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/STSSL/data_preprocessing/detection_box_generate_ScanNet.py', '--start', '200', '--end', '400', '--segment_save_path', segment_save_path, '--box_save_path', box_save_path], stdout = box_file )
    result_all.append(result)
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/STSSL/data_preprocessing/detection_box_generate_ScanNet.py', '--start', '400', '--end', '600', '--segment_save_path', segment_save_path, '--box_save_path', box_save_path], stdout = box_file )
    result_all.append(result)
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/STSSL/data_preprocessing/detection_box_generate_ScanNet.py', '--start', '600', '--end', '800', '--segment_save_path', segment_save_path, '--box_save_path', box_save_path], stdout = box_file )
    result_all.append(result)
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/STSSL/data_preprocessing/detection_box_generate_ScanNet.py', '--start', '800', '--end', '1000', '--segment_save_path', segment_save_path, '--box_save_path', box_save_path], stdout = box_file )
    result_all.append(result)
    result = subprocess.Popen(['python', '/home/wuyanhao/WorkSpace/STSSL/data_preprocessing/detection_box_generate_ScanNet.py', '--start', '1000', '--end', '1202', '--segment_save_path', segment_save_path, '--box_save_path', box_save_path], stdout = box_file )
    result_all.append(result)
    while(True):
      count = 0
      for i_count in range(len(result_all)):
        if result_all[i_count].poll() is None:
          count = 0 
          break;
        else:
          count+=1
      if count == len(result_all):
        break;
    print(time.strftime('%Y-%m-%d %H:%M:%S')) #结构化输出当前的时间
    print("Finish Box generating")
    file.flush()


file.close()
box_file.close()
generate_file.close()
segment_file.close()
train_file.close()
sys.stdout = oldstdout
