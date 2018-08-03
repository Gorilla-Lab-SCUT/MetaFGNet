##############################################################################
#
# All the codes about the model constructing should be kept in the folder ./models/
# All the codes about the data process should be kept in the folder ./data/
# The file ./opts.py stores the options.
# The file ./trainer.py stores the training and test strategy
# The ./main.py should be simple
#
##############################################################################
import os
import json
import shutil
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
import random
import numpy as np
import torch.backends.cudnn as cudnn
from models.resnet import resnet  # The model construction
from trainer import train  # For the training process
from trainer import validate  # For the validate (test) process
from opts import opts  # The options for the project
from data.prepare_data import generate_dataloader  # Prepare the data and dataloader
import ipdb

best_prec1 = 0

def main():
    global args, best_prec1
    args = opts()
    # ipdb.set_trace()
    # args = parser.parse_args()
    model_source, model_target = resnet(args)
    # define-multi GPU
    model_source = torch.nn.DataParallel(model_source).cuda()
    model_target = torch.nn.DataParallel(model_target).cuda()
    print('the memory id should be same for the shared feature extractor:')
    print(id(model_source.module.resnet_conv))   # the memory is shared here
    print(id(model_target.module.resnet_conv))
    print('the memory id should be different for the different classifiers:')
    print(id(model_source.module.fc))  # the memory id shared here.
    print(id(model_target.module.fc))
    # define loss function(criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    np.random.seed(1)  ### fix the random data.
    random.seed(1)
    # optimizer = torch.optim.SGD(model.parameters(),
    # To apply different learning rate to different layer
    if args.meta_sgd:
        meta_train_lr = []
        for param in model_target.parameters():
            meta_train_lr.append(torch.FloatTensor(param.data.size()).fill_(args.meta_train_lr).cuda())
    if args.pretrained:
        print('the pretrained setting of optimizer')
        if args.auxiliary_dataset == 'imagenet':
            optimizer = torch.optim.SGD([
                {'params': model_source.module.resnet_conv.parameters(), 'name': 'pre-trained'},
                {'params': model_source.module.fc.parameters(), 'name': 'pre-trained'},
                {'params': model_target.module.fc.parameters(), 'name': 'new-added'},
            ],
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args.auxiliary_dataset == 'l_bird':
            optimizer = torch.optim.SGD([
                {'params': model_source.module.resnet_conv.parameters(), 'name': 'pre-trained'},
                {'params': model_source.module.fc.parameters(), 'name': 'pre-trained'},
                {'params': model_target.module.fc.parameters(), 'name': 'new-added'},
            ],
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    else:
        print('the from scratch setting of optimizer')
        optimizer = torch.optim.SGD([
            {'params': model_source.module.resnet_conv.parameters(), 'name': 'new-added'},
            {'params': model_source.module.fc.parameters(), 'name': 'new-added'},
            {'params': model_target.module.fc.parameters(), 'name': 'new-added'},
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    #optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # raise ValueError('the resume function is not finished')
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.meta_sgd:
                meta_train_lr = checkpoint['meta_train_lr']
            best_prec1 = checkpoint['best_prec1']
            model_source.load_state_dict(checkpoint['source_state_dict'])
            model_target.load_state_dict(checkpoint['target_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}'(epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from is not exited', args.resume)

    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    cudnn.benchmark = True
    # process the data and prepare the dataloaders.
    dataloader_returned = generate_dataloader(args)
    dataloader_number_returned = len(dataloader_returned)
    print('the number of dataloader number returned is: ', dataloader_number_returned)
    if dataloader_number_returned != 2:
        train_loader_source, val_loader_source, train_loader_target, val_loader_target = dataloader_returned
    else:
        train_loader_target, val_loader_target = dataloader_returned
        train_loader_source = None
    # train_loader, val_loader = generate_dataloader(args)
    # test only
    if args.test_only:
        if dataloader_number_returned == 2:
            validate(None, val_loader_target, model_source, model_target, criterion, 0, args)
        else:
            validate(val_loader_source, val_loader_target, model_source, model_target, criterion, 0, args)
        # if args.auxiliary_dataset == 'imagenet':
        #     validate(val_loader_source, val_loader_target, model_source, model_target, criterion, 0, args)
        # else:
        #     validate(None, val_loader_target, model_source, model_target, criterion, 0, args)
        return

    print('begin training')
    if train_loader_source:
        train_loader_source_batch = enumerate(train_loader_source)
    else:
        train_loader_source_batch = None
    train_loader_target_batch = enumerate(train_loader_target)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if args.meta_sgd:
            train_loader_source_batch, train_loader_target_batch, meta_train_lr = train(train_loader_source, train_loader_source_batch, train_loader_target,train_loader_target_batch, model_source, model_target, criterion, optimizer, epoch, args, meta_train_lr)
        else:
            train_loader_source_batch, train_loader_target_batch = train(train_loader_source, train_loader_source_batch, train_loader_target,train_loader_target_batch, model_source, model_target, criterion, optimizer, epoch, args, None)
        # train(train_loader, model, criterion, optimizer, epoch, args)
        # evaluate on the val data
        if (epoch + 1) % args.test_freq == 0 or (epoch + 1) % args.epochs == 0:
            if dataloader_number_returned == 2:
                prec1 = validate(None, val_loader_target, model_source, model_target, criterion, epoch, args)
            else:
                prec1 = validate(val_loader_source, val_loader_target, model_source, model_target, criterion, epoch, args)
            # prec1 = 1
            # record the best prec1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                log = open(os.path.join(args.log, 'log.txt'), 'a')
                log.write('     \nTarget_T1 acc: %3f' % (best_prec1))
                log.close()
            if args.meta_sgd:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'meta_train_lr': meta_train_lr,
                    'arch': args.arch,
                    'source_state_dict': model_source.state_dict(),
                    'target_state_dict': model_target.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args, epoch)
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'source_state_dict': model_source.state_dict(),
                    'target_state_dict': model_target.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args, epoch + 1)


def save_checkpoint(state, is_best, args, epoch):
    filename = str(epoch) + 'checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()





