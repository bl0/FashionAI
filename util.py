#!/usr/bin/env python
# encoding: utf-8

import torch
import shutil

def write_results(df_load, test_np, save_path):
    result = []

    for i, row in df_load.iterrows():
        tmp_list = test_np[i]
        tmp_result = ''
        for tmp_ret in tmp_list:
            tmp_result += '{:.4f};'.format(tmp_ret)

        result.append(tmp_result[:-1])

    df_load['result'] = result
    df_load.to_csv(save_path, header=None, index=False)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='best.pth.tar'):
    if not is_best:
        return
    torch.save(state, filename)
    shutil.copyfile(filename, best_filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    _, pred = output.max(1)
    correct = pred.eq(target).float()
    return correct.sum(0).mul_(100.0 / target.size(0))
