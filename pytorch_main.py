#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import time

import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from util import save_checkpoint, AverageMeter, accuracy, accuracy_all, write_results
from fashionai_dataset import FashionAIDataset, FashionAITestDataset, classes

from pprint import pprint
import warnings
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")


model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch Fashion AI')
parser.add_argument('--data', type=str, help='path to dataset', metavar='DIR',
                    default='/home/liujintao/runspace/FashionAI/data/fashionAI_attributes_train_20180222')
parser.add_argument('--test-data', type=str, help='path to dataset', metavar='DIR',
                    default='/home/liujintao/runspace/FashionAI/data/fashionAI_attributes_test_a_20180222')
parser.add_argument('--cur_class_idx', default=0, type=int, help='The index of label to classify, -1 for all')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--tr_input_size', default=224, type=int,
                    metavar='N', help='input size for training (default: 224)')
parser.add_argument('--te_input_size', default=224, type=int,
                    metavar='N', help='input size for testing (default: 224)')
parser.add_argument('--te_resize_size', default=256, type=int,
                    metavar='N', help='resize size for testing (default: 256)')
parser.add_argument('--ten_crop', dest='ten_crop', action='store_true',
                    help='use ten crop for testing data')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-vb', '--val-batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpus', default='5', type=str, metavar='GPU', help='id of gpu')
parser.add_argument('--inference', dest='inference', action='store_true',  help='inference test data')
parser.add_argument('--result_path', default='./results', type=str, metavar='PATH', help='folder to save result')
parser.add_argument('--model_save_path', default='./models', type=str, metavar='PATH', help='folder to save models')
parser.add_argument('--tensorboard_log_path', default='./tensorboard_log', type=str, metavar='PATH', help='folder to place tensorboard logs')

args = parser.parse_args()
best_prec1 = 0
class_name = None
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

classes_len = {'collar_design_labels':0,
                  'neckline_design_labels':5,
                  'skirt_length_labels':15,
                  'sleeve_length_labels':21,
                  'neck_design_labels':30,
                  'coat_length_labels':35,
                  'lapel_design_labels':43,
                  'pant_length_labels':48}

# for tensorboard
name = '{}_{}_tri{}_ter{}_tei{}.pth.tar'.format(args.arch, args.cur_class_idx, args.tr_input_size, args.te_resize_size, args.te_input_size)
if args.ten_crop:
    name = '{}_{}'.format(args.arch, args.cur_class_idx)
writer = SummaryWriter(os.path.join(args.tensorboard_log_path, name))

# writer = SummaryWriter(args.tensorboard_log_path)

global_train_step = 0

def load_data():
    global class_name
    if args.cur_class_idx == -1:
        class_name = 'all'
        traindir = os.path.join(args.data, 'train', 'all_labels.csv')
        valdir = os.path.join(args.data, 'val', 'all_labels.csv')
        #print("not supported yet")
    else:
        class_name = classes[args.cur_class_idx]
        traindir = os.path.join(args.data, 'train', class_name + '.csv')
        valdir = os.path.join(args.data, 'val', class_name + '.csv')
    testdir = os.path.join(args.test_data, 'Tests', 'question.csv')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = FashionAIDataset(
        traindir, args.data,
        class_name=class_name,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(args.tr_input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_transform = transforms.Compose([
            transforms.Resize(args.te_resize_size),
            transforms.TenCrop(args.te_input_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
	]) if args.ten_crop else transforms.Compose([
            transforms.Resize(args.te_resize_size),
            transforms.CenterCrop(args.te_input_size),
            transforms.ToTensor(),
            normalize,
	])
    val_dataset = FashionAIDataset(valdir, args.data, class_name=class_name, transform=val_transform)
    test_dataset = FashionAITestDataset(testdir, args.test_data, class_name=class_name, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader

def build_model(n_class):
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = torch.nn.Linear(2048, n_class)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            global best_prec1
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    return model, optimizer

def main():
    pprint(vars(args))


    global best_prec1
    train_loader, val_loader, test_loader = load_data()
    model, optimizer = build_model(train_loader.dataset.n_class)
    criterion = nn.CrossEntropyLoss().cuda()

    if args.inference:
        inference(test_loader, model)
        return

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save checkpoint
        state = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }
        best_path = os.path.join(args.model_save_path, 'best_models')
        if not os.path.exists(best_path):
            os.makedirs(best_path)
        filename = "{}/{}_{}_{}_checkpoint.pth.tar".format(args.model_save_path, args.arch, class_name, prec1)
        best_filename = '{}/{}_{}.pth.tar'.format(best_path, args.arch, class_name)

        save_checkpoint(state, is_best, filename, best_filename)

        # tensorboad record
        writer.add_scalar('val_prec', prec1, global_train_step)
        writer.file_writer.flush()

        # precision
        print(' * Prec {prec:.3f}, bestPrec1 = {best_prec1:.3f}'.format(prec=prec1, best_prec1=best_prec1))

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


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, idx) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        if args.arch == 'inception_v3':
            output = output[0]

        # new weighted softmax loss
        loss = weighted_softmax_loss(output, target_var, idx)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        prec.update(accuracy_all(output.data, target, idx)[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print
        if (epoch == 0 and i < 50) or i % args.print_freq == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {prec.val:.3f} ({prec.avg:.3f})'.format(
                   epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, prec=prec))

        # tensorboard record
        global global_train_step
        writer.add_scalar('loss', losses.val, global_train_step)
        writer.add_scalar('prec', prec.val, global_train_step)
        global_train_step += 1

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    prec = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # TODO
    for i, (input, target, idx) in enumerate(val_loader):
        target = target.cuda(async=True)

        if args.ten_crop:
            bs, ncrop, c, h, w = input.size()
            input = input.view(-1, c, h, w)
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        output = model(input_var)
        if args.ten_crop:
            output = output.view(bs, ncrop, -1).mean(1)
        target_var = torch.autograd.Variable(target, volatile=True)

        #loss = criterion(output, target_var)
        loss = weighted_softmax_loss(output, target_var, idx)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        prec.update(accuracy_all(output.data, target, idx)[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        a = len(val_loader.batch_sampler)
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec {prec.val:.3f} ({prec.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time,
                loss=losses, prec=prec))

    return prec.avg


def inference(test_loader, model):
    # switch to evaluate mode
    model.eval()
    results = []

    for i, (input, idx) in enumerate(test_loader):
        if args.ten_crop:
            bs, ncrop, c, h, w = input.size()
            input = input.view(-1, c, h, w)

        input_var = torch.autograd.Variable(input, volatile=True)
        output = model(input_var)

        if args.ten_crop:
            output = output.view(bs, ncrop, -1).mean(1)

        real_output = torch.mul(output, torch.autograd.Variable(idx).float().cuda())

        for i in range(real_output.size()[0]):
            append_value = torch.nn.functional.softmax(real_output[i][real_output[i]!= 0].float())
            results.append(append_value.data.cpu().numpy())

    save_path = os.path.join(args.result_path, class_name+".csv")
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    write_results(test_loader.dataset.df_load, results, save_path)

    print('Inference done')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
