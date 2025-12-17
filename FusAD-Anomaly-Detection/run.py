"""
FusAD-Anomaly-Detection: Joint Adaptive Frequency-Aware Network for Time Series Anomaly Detection
主运行脚本
"""
import argparse
import os
import torch
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='FusAD-Anomaly-Detection')

# basic config
parser.add_argument('--task_name', type=str, default='anomaly_detection',
                    help='task name: anomaly_detection')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='FusAD',
                    help='model name: FusAD')

# data loader
parser.add_argument('--data', type=str, required=True, default='MSL', help='dataset type: MSL, SMAP, SMD, PSM, SWAT')
parser.add_argument('--root_path', type=str, default='./data/MSL/', help='root path of the data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# anomaly detection task
parser.add_argument('--seq_len', type=int, default=100, help='input sequence length / window size')
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# model define
parser.add_argument('--enc_in', type=int, default=55, help='encoder input size / number of features')
parser.add_argument('--c_out', type=int, default=55, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers / depth')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout value')
parser.add_argument('--patch_size', type=int, default=8, help='size of patches')
parser.add_argument('--stride', type=int, default=4, help='stride for patching')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Anomaly_Detection

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = 'FusAD_{}_{}_{}_sl{}_dm{}_el{}_ps{}_st{}_{}'.format(
            args.task_name,
            args.model_id,
            args.data,
            args.seq_len,
            args.d_model,
            args.e_layers,
            args.patch_size,
            args.stride,
            args.des)

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = 'FusAD_{}_{}_{}_sl{}_dm{}_el{}_ps{}_st{}_{}'.format(
        args.task_name,
        args.model_id,
        args.data,
        args.seq_len,
        args.d_model,
        args.e_layers,
        args.patch_size,
        args.stride,
        args.des)

    exp = Exp(args)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
