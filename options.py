#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import pretrainedmodels

model_names = pretrainedmodels.model_names

parser = argparse.ArgumentParser(description='PyTorch Fashion AI')

# model
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet152',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names))
parser.add_argument('--tr_input_size', default=224, type=int,
                    metavar='N', help='input size for training (default: 224)')
parser.add_argument('--te_input_size', default=224, type=int,
                    metavar='N', help='input size for testing (default: 224)')
parser.add_argument('--te_resize_size', default=256, type=int,
                    metavar='N', help='resize size for testing (default: 256)')
parser.add_argument('--crop', type=str, default='ten', choices=['ten', 'no', 'center'],
                    help='if no crop, val_batch_size must be 1')

pretrained_parser = parser.add_mutually_exclusive_group(required=False)
pretrained_parser.add_argument('--pretrained', dest='pretrained', action='store_true')
pretrained_parser.add_argument('--no_pretrained', dest='pretrained', action='store_false')
parser.set_defaults(pretrained=True)

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--inference', dest='inference', action='store_true',  help='inference test data')

# training and optimizer
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-vb', '--val-batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--opt', default='SGD', type=str.lower, metavar='OPT',
                    help='optimizer')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--decay-rate', '--dr', default=0.1, type=float,
                    metavar='Decay Rate', help='Decay rate (default: 0.1)')
parser.add_argument('--decay-type', '--dt', default='multistep', type=str,
                    metavar='Decay Type', help='Decay type, cosine or multistep')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='dropout rate, default 0, minus means linear gradual dropout')
parser.add_argument('--no-norm', action='store_true',
                    help='use batch norm before 1st conv')

# data and io related
parser.add_argument('--data', type=str, help='path to dataset', metavar='DIR',
                    default='/runspace/liubin/tianchi2018_fashion-tag/data/fashionAI_attributes_train_20180222')
parser.add_argument('--test-data', type=str, help='path to dataset', metavar='DIR',
                    default='/runspace/liubin/tianchi2018_fashion-tag/data/fashionAI_attributes_test_a_20180222')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--result-path', default='./results', type=str,
                    metavar='PATH', help='folder to save result')
parser.add_argument('--model-save-path', default='./models', type=str,
                    metavar='PATH', help='folder to save models')
parser.add_argument('--tensorboard-log-path', default='./tensorboard_log', type=str,
                    metavar='PATH', help='folder to place tensorboard logs')

parser.add_argument('--explain', default='', type=str)

# others
parser.add_argument('--cur_class_idx', default=-1, type=int, help='The index of label to classify, -1 for all')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpus', default='5', type=str, metavar='GPU', help='id of gpu')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
