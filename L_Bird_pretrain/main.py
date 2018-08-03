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
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models.resnet import resnet  # The model construction
from trainer import train  # For the training process
from trainer import validate  # For the validate (test) process
from opts import opts  # The options for the project
from data.prepare_data import generate_dataloader  # Prepare the data and dataloader

best_prec1 = 0

def main():
    global args, best_prec1
    args = opts()
    # args = parser.parse_args()
    model = resnet(args.arch, args.pretrain, args)
    # define-multi GPU
    model = torch.nn.DataParallel(model).cuda()
    #print(model)
    # define loss function(criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(),
    # To apply different learning rate to different layer
    #print(model.module)
    optimizer = torch.optim.SGD([
        {'params': model.module.conv1.parameters(), 'name': 'pre-trained'},
        {'params': model.module.bn1.parameters(), 'name': 'pre-trained'},
        {'params': model.module.layer1.parameters(), 'name': 'pre-trained'},
        {'params': model.module.layer2.parameters(), 'name': 'pre-trained'},
        {'params': model.module.layer3.parameters(), 'name': 'pre-trained'},
        {'params': model.module.layer4.parameters(), 'name': 'pre-trained'},
        {'params': model.module.fc.parameters(), 'lr': args.lr*10, 'name': 'new-added'}
    ],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}'(epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from is not exited', args.resume)
    else:
        if not os.path.isdir(args.log):
            os.makedirs(args.log)
        log = open(os.path.join(args.log, 'log.txt'), 'w')
        state = {k: v for k, v in args._get_kwargs()}
        log.write(json.dumps(state) + '\n')
        log.close()

    cudnn.benchmark = True
    # process the data and prepare the dataloaders.
    train_loader, val_loader = generate_dataloader(args)
    #test only
    if args.test_only:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        # evaluate on the val data
        prec1 = validate(val_loader, model, criterion, epoch, args)
        # record the best prec1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            log = open(os.path.join(args.log, 'log.txt'),'a')
            log.write("      best result is %3f" %(prec1))
            log.close()
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, epoch, is_best, args)


def save_checkpoint(state,epoch, is_best, args):
    filename = str(epoch) +  '_checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()





