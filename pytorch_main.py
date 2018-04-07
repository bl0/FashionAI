#!/usr/bin/env python
# encoding: utf-8

import os
import time
import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from util import save_checkpoint
from util import AverageMeter
from util import accuracy_all
from util import write_results
from util import weighted_softmax_loss
from util import MAP
from options import args
from fashionai_dataset import FashionAIDataset
from fashionai_dataset import FashionAITestDataset
from fashionai_dataset import classes

from pprint import pprint
import warnings
from tensorboardX import SummaryWriter
import pretrainedmodels

warnings.filterwarnings("ignore")

best_mAP = 0
best_prec = 0
dropout_ratio = 0.2

# for tensorboard
name = '{args.arch}_{args.cur_class_idx}_{args.opt}_{args.decay_type}_lr_{args.lr}_wd_{args.weight_decay}_dropout_{dropout_ratio}'.format(args=args, dropout_ratio=dropout_ratio)
writer = SummaryWriter(os.path.join(args.tensorboard_log_path, name))

global_train_step = 0

def load_data(opts):
    if args.cur_class_idx == -1:
        class_name = 'all'
        traindir = os.path.join(args.data, 'train', 'all_labels.csv')
        valdir = os.path.join(args.data, 'val', 'all_labels.csv')
    else:
        class_name = classes[args.cur_class_idx]
        traindir = os.path.join(args.data, 'train', class_name + '.csv')
        valdir = os.path.join(args.data, 'val', class_name + '.csv')
    testdir = os.path.join(args.test_data, 'Tests', 'question.csv')

    normalize = transforms.Normalize(mean=opts.mean, std=opts.std)

    train_dataset = FashionAIDataset(
        traindir, args.data,
        class_name=class_name,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(max(opts.input_size)), # TODO: if input size not fixed
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.crop == 'ten':
        resize = transforms.Resize(int(args.test_input_size / 0.875))
        crop = transforms.TenCrop(args.test_input_size)
        to_tensor = transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        norm = transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))

        val_transform = transforms.Compose([resize, crop, to_tensor, norm])
    elif args.crop == 'center':
        resize = transforms.Resize(args.test_input_size)
        crop = transforms.CenterCrop(args.test_input_size)
        to_tensor = transforms.ToTensor()
        norm = normalize

        val_transform = transforms.Compose([resize, crop, to_tensor, norm])
    elif args.crop == 'no':
        resize = transforms.Resize(args.test_input_size)
        to_tensor = transforms.ToTensor()
        norm = normalize

        val_transform = transforms.Compose([resize, to_tensor, norm])

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

def build_model():
    n_classes = [5, 10, 6, 9, 5, 8, 5, 6]
    if args.cur_class_idx == -1:
        n_class = sum(n_classes)
    else:
        n_class = n_classes[cur_class_idx]

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = pretrainedmodels.__dict__[args.arch](num_classes=1000)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained=None)

    if args.arch.startswith('dpn'):
        model.classifier = nn.Conv2d(model.classifier.in_channels, n_class, kernel_size=1, bias=True)
        # last_conv = nn.Conv2d(model.classifier.in_channels, n_class, kernel_size=1, bias=True)
        # model.classifier = nn.Sequential(last_conv, nn.Dropout(dropout_ratio, inplace=True))
        feat_param = list(model.features.named_parameters())
        cls_param = list(model.classifier.named_parameters())
    elif args.arch.startswith('resnet'):
        model.avgpool = nn.AdaptiveAvgPool2d(1)

        # no dropout
        # model.last_linear = torch.nn.Linear(model.last_linear.in_features, n_class)

        # dropout
        last_linear = torch.nn.Linear(model.last_linear.in_features, n_class)
        model.last_linear = nn.Sequential(last_linear, nn.Dropout(dropout_ratio, inplace=True))

        cls_param = list(model.last_linear.named_parameters())
        feat_param = [param for param in model.named_parameters() if not param[0].startswith('last_linear')]
    else:
        print('not supported yet')
        return

    # define optimizer
    args.param_groups = [
        {'params': [param for name, param in feat_param if name[-4:] == 'bias'],
         'lr': 2},
        {'params': [param for name, param in feat_param if name[-4:] != 'bias'],
         'lr': 1, 'weight_decay': args.weight_decay},
        {'params': [param for name, param in cls_param if name[-4:] == 'bias'],
         'lr': 2},
        {'params': [param for name, param in cls_param if name[-4:] != 'bias'],
         'lr': 1, 'weight_decay': args.weight_decay},
    ]
    if args.opt== "adam":
        optimizer = torch.optim.Adam([i.copy() for i in args.param_groups])
    elif args.opt == 'amsgrad':  # need pytorch master version
        optimizer = torch.optim.Adam([i.copy() for i in args.param_groups], amsgrad=True)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD([i.copy() for i in args.param_groups], momentum=args.momentum, nesterov=True) # TODO
    else:
        print('Not supported yet')

    # data parallel for multi-gpu
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            global best_mAP, best_prec
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mAP = checkpoint['best_mAP']
            best_prec = checkpoint.get('best_prec', 0)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    return model, optimizer

def main():
    pprint(vars(args))

    global best_mAP, best_prec
    model, optimizer = build_model()
    train_loader, val_loader, test_loader = load_data(model.module)
    criterion = nn.CrossEntropyLoss().cuda()

    if args.inference:
        inference(test_loader, model)
        return

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        mAP, prec = validate(val_loader, model, criterion)

        # remember best mAP and save checkpoint
        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        best_prec = max(prec, best_prec)

        # save checkpoint
        state = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_mAP': best_mAP,
            'best_prec': best_prec,
            'optimizer' : optimizer.state_dict(),
        }
        best_path = os.path.join(args.model_save_path, 'best_models')
        if not os.path.exists(best_path):
            os.makedirs(best_path)
        class_name = train_loader.dataset.class_name
        filename = "{}/{}_{}_{}_checkpoint.pth.tar".format(args.model_save_path, args.arch, class_name, mAP)
        best_filename = '{}/{}_{}.pth.tar'.format(best_path, args.arch, class_name)
        save_checkpoint(state, is_best, filename, best_filename)

        # tensorboad record
        writer.add_scalar('val_prec', prec, global_train_step)
        writer.add_scalar('val_mAP', mAP, global_train_step)
        writer.add_scalar('val_prec_epoch', prec, epoch)
        writer.add_scalar('val_mAP_epoch', mAP, epoch)
        writer.file_writer.flush()

        # print mAP
        print(' * best mAP = {best_mAP:.3f}, best prec = {best_prec:.3f}'.format(best_mAP=best_mAP, best_prec=best_prec))

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

        if args.opt == 'sgd':
            adjust_learning_rate(optimizer, epoch, n_batch=len(train_loader), method=args.decay_type)

        # convert to variable
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

    output_all, target_all, idx_all = [], [], []
    end = time.time()
    for i, (input, target, idx) in enumerate(val_loader):
        target_all.append(target)
        idx_all.append(idx)

        target = target.cuda(async=True)

        if args.crop == 'ten':
            bs, ncrop, c, h, w = input.size()
            input = input.view(-1, c, h, w)
        input_var = torch.autograd.Variable(input, volatile=True)

        # compute output
        output = model(input_var)
        if args.crop == 'ten':
            output = output.view(bs, ncrop, -1).mean(1)
        output_all.append(output.cpu().data.numpy())
        target_var = torch.autograd.Variable(target, volatile=True)

        #loss = criterion(output, target_var)
        loss = weighted_softmax_loss(output, target_var, idx)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        prec.update(accuracy_all(output.data, target, idx)[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Val: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec {prec.val:.3f} ({prec.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time,
                loss=losses, prec=prec))

    output_all = np.concatenate(output_all)
    target_all = np.concatenate(target_all)
    idx_all = np.concatenate(idx_all)
    mAP = MAP(output_all, target_all, idx_all)

    print(' * Prec {prec:.3f}, mAP = {mAP:.3f}'.format(prec=prec.avg, mAP=mAP))
    return mAP, prec.avg


def inference(test_loader, model):
    # switch to evaluate mode
    model.eval()

    results = []
    for i, (input, idx) in enumerate(test_loader):
        if args.crop == 'ten':
            bs, ncrop, c, h, w = input.size()
            input = input.view(-1, c, h, w)

        input_var = torch.autograd.Variable(input, volatile=True)
        output = model(input_var)

        if args.crop == 'ten':
            output = output.view(bs, ncrop, -1).mean(1)

        real_output = torch.mul(output, torch.autograd.Variable(idx).float().cuda())

        for j in range(real_output.size()[0]):
            append_value = torch.nn.functional.softmax(real_output[j][real_output[j]!= 0].float())
            results.append(append_value.data.cpu().numpy())

        if i % args.print_freq == 0:
            print('Inference: [{0}/{1}]'.format(i, len(test_loader)))


    # write results to file
    save_path = os.path.join(args.result_path, test_loader.dataset.class_name+".csv")
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    write_results(test_loader.dataset.df_load, results, save_path)

    print('Inference done')

def adjust_learning_rate(optimizer, epoch, n_batch=None, method='cosine'):
    if method == 'cosine':
        cosine = lambda x: 0.5 * (1 + math.cos(math.pi * x))
        lr = args.lr * cosine(global_train_step / (args.epochs * n_batch))
    elif method == 'multistep':
        lr, decay_rate = args.lr, args.decay_rate
        if epoch >= args.epochs * 0.75:
            lr *= decay_rate**2
        elif epoch >= args.epochs * 0.5:
            lr *= decay_rate
        # lr = args.lr * args.decay_rate ** (epoch // 9)

    writer.add_scalar('lr', lr, global_train_step)
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * args.param_groups[i]['lr']


if __name__ == '__main__':
    main()
