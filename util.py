#!/usr/bin/env python
# encoding: utf-8

import torch
import shutil
import numpy as np

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
    _, pred = output.max(1)
    correct = pred.eq(target).float()
    return correct.sum(0).mul_(100.0 / target.size(0))

def accuracy_all(output, target, idx):
    idx_cum = np.cumsum(idx, 1)
    idx_1 = np.array(np.where(idx_cum == 1))
    real_output = torch.mul(output, idx.float().cuda())

    _, pred = real_output.max(1)
    pred = pred - torch.from_numpy(idx_1[1]).cuda()

    correct = pred.eq(target).float()
    return correct.sum(0).mul_(100.0 / target.size(0))

def weighted_softmax_loss(output, target, weight):
    output_e = torch.exp(output)

    weighted_output_e = torch.mul(output_e, torch.autograd.Variable(weight).float().cuda())
    output_sum = torch.sum(weighted_output_e, dim=1)

    # weight is like [0,0,0,1,1,1,0,0,0]
    # find the first value 1 for each row
    # idx -> N*2, including the indexes
    weight_cum = np.cumsum(weight, 1)
    idx = np.array(np.where(weight_cum == 1))

    # target -> N*1, indicating which is the label
    idx[1] = idx[1] + target.cpu().data.numpy()

    # loss = - log( e^y / sum) = log sum - y
    output_t = output[idx]
    final_loss = torch.mean(torch.log(output_sum) - output_t)
    return final_loss

def np_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def MAP(output, target, idx):
    occur = [np.where(row == 1)[0] for row in idx]
    first_and_last = [(row[0], row[-1]) for row in occur]
    first_np = np.array(first_and_last)[:, 0]
    APs = []
    for (first, last) in set(first_and_last):
        output_c = output[first_np == first, first: last+1]
        target_c = target[first_np == first]
        probs = np_softmax(output_c)

        # get sorted index by max attribute prob
        sorted_idx = probs.max(axis=-1).argsort()
        # get sorted predicted attrivute value
        sorted_attr_value = probs.argmax(axis=-1)[sorted_idx]

        weight = np.arange(len(sorted_idx)) + 1

        APs.append(np.sum((sorted_attr_value == target_c[sorted_idx]) * weight) / np.sum(weight))
    print('* APs: ' + str(APs))
    return np.mean(APs) * 100
