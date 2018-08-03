import time
import torch
import os
import copy
import ipdb


def train(train_loader_source, train_loader_source_batch, train_loader_target, train_loader_target_batch, model_source, model_target, criterion, optimizer, epoch, args, meta_train_lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_source = AverageMeter()
    top1_source = AverageMeter()
    top5_source = AverageMeter()
    losses_target = AverageMeter()
    losses_real = AverageMeter()
    top1_target = AverageMeter()
    top5_target = AverageMeter()
    model_source.train()
    model_target.train()

    adjust_learning_rate(optimizer, epoch, args)
    end = time.time()

    # for i, (input_source, target_source) in enumerate(train_loader_source):  # the iterarion in the source dataset.
    # prepare the data for the model forward and backward
    if train_loader_source:
        try:
            (input_source, target_source) = train_loader_source_batch.__next__()[1]
        except StopIteration:
            train_loader_source_batch = enumerate(train_loader_source)
            (input_source, target_source) = train_loader_source_batch.__next__()[1]
        target_source = target_source.cuda(async=True)
        input_source_var = torch.autograd.Variable(input_source)
        target_source_var = torch.autograd.Variable(target_source)

    # target_l = enumerate(train_loader_target)  # !has been verified no overlap!!
    try:
        temp_train = train_loader_target_batch.__next__()
        (input_target_train, target_target_train) = temp_train[1]
    except StopIteration:
        train_loader_target_batch = enumerate(train_loader_target)
        temp_train = train_loader_target_batch.__next__()
        (input_target_train, target_target_train) = temp_train[1]

    try:
        temp_mtrain = train_loader_target_batch.__next__()
        (input_target_mtrain, target_target_mtrain) = temp_mtrain[1]
    except StopIteration:
        train_loader_target_batch = enumerate(train_loader_target)
        temp_mtrain = train_loader_target_batch.__next__()
        (input_target_mtrain, target_target_mtrain) = temp_mtrain[1]

    data_time.update(time.time() - end)
    target_target_train = target_target_train.cuda(async=True)
    input_target_train_var = torch.autograd.Variable(input_target_train)
    target_target_train_var = torch.autograd.Variable(target_target_train)
    target_target_mtrain = target_target_mtrain.cuda(async=True)
    input_target_mtrain_var = torch.autograd.Variable(input_target_mtrain)
    target_target_mtrain_var = torch.autograd.Variable(target_target_mtrain)

    model_target_temp = copy.deepcopy(model_target)
    model_target_temp.train()
    
    if not args.meta_sgd:
        optimizer_target_temp = torch.optim.SGD([
            {'params': model_target_temp.module.resnet_conv.parameters(), 'name': 'pre-trained'},
            {'params': model_target_temp.module.fc.parameters(), 'name': 'new-added'}
            ],
            lr=args.meta_train_lr,
            momentum=0.0,
            weight_decay=0.0)
        meta_train_lr = adjust_meta_train_learning_rate(optimizer_target_temp, epoch, args)
        grad_for_task_train = []  # just retain the gradient with model inputing a meta-train batch before the first update
    else:
        if args.auxiliary_dataset == 'imagenet' and not args.pretrained:
            raise ValueError('the process is not finished')
        grads_for_task_train = [] # retain the gradients with model inputing a meta-train batch before the first and last update
    # if args.second_order_grad:
    #     second_order_grads_for_task_train = [] # retain all the second-order gradients with model inputing a meta-train batch before each update
    grad_for_task_mtrain = []

    for k in range(args.num_updates_for_gradient):
        # ipdb.set_trace()
        output_target_train = model_target_temp(input_target_train_var)
        loss_target_train = criterion(output_target_train, target_target_train_var)
        model_target_temp.zero_grad()
        loss_target_train.backward()  # what accumulate into .grad is the same as grad_params

        if not args.meta_sgd:
            if k == 0:
                temp_grad = []
                for param in model_target_temp.parameters():
                    temp_grad.append(param.grad.data.clone())
                grad_for_task_train.extend(temp_grad)
            ### update model parameters using learned learning rate
            optimizer_target_temp.step()
        else:
            if k == 0 or k == (args.num_updates_for_gradients - 1):
                temp_grad = []
                for param in model_target_temp.parameters():
                    temp_grad.append(param.grad.data.clone())
                grads_for_task_train.append(temp_grad)
            ### update model parameters using learned learning rate
            meta_train_update(model_target_temp, meta_train_lr)

    output_target = model_target_temp(input_target_mtrain_var)
    loss_target = criterion(output_target, target_target_mtrain_var)
    model_target_temp.zero_grad()
    loss_target.backward()

    temp_grad = []
    for param in model_target_temp.parameters():
        temp_grad.append(param.grad.data.clone())
    grad_for_task_mtrain.extend(temp_grad)
    
    grad_of_meta_test = copy.deepcopy(grad_for_task_mtrain)
    
    if args.second_order_grad:
        for m in range(args.num_updates_for_gradient):
            model_target_temp = copy.deepcopy(model_target)
            model_target_temp.train()
            if args.pretrained:
                optimizer_target_temp = torch.optim.SGD([
                {'params': model_target_temp.module.resnet_conv.parameters(), 'name': 'pre-trained'},
                {'params': model_target_temp.module.fc.parameters(), 'name': 'new-added'}
                ],
                lr=meta_train_lr,
                momentum=0.0,
                weight_decay=0.0)
            else:
                optimizer_target_temp = torch.optim.SGD([
                {'params': model_target_temp.module.resnet_conv.parameters(), 'name': 'new-added'},
                {'params': model_target_temp.module.fc.parameters(), 'name': 'new-added'}
                ],
                lr=meta_train_lr,
                momentum=0.0,
                weight_decay=0.0)
            for k in range(args.num_updates_for_gradient - m):
                output_target_train = model_target_temp(input_target_train_var)
                loss_target_train = criterion(output_target_train, target_target_train_var)
                model_target_temp.zero_grad()
                if k == args.num_updates_for_gradient - m - 1:
                    grad_params = torch.autograd.grad(loss_target_train, model_target_temp.parameters(), create_graph=True)
                    grad_for_task_mtrain = compute_second_order_grad(args, meta_train_lr, grad_params, grad_for_task_mtrain, model_target_temp)
                    break
                
                loss_target_train.backward() # what accumulate into .grad is the same as grad_params
    
                if not args.meta_sgd:
                    ### update model parameters using learned learning rate
                    optimizer_target_temp.step()
                else:
                    ### update model parameters using learned learning rate
                    meta_train_update(model_target_temp, meta_train_lr)

    # calculate for the target data#####################################################
    # mesure accuracy and record loss
    prec1, prec5 = accuracy(output_target.data, target_target_mtrain, topk=(1, 5))
    losses_target.update(loss_target.data[0], input_target_mtrain.size(0))
    top1_target.update(prec1[0], input_target_mtrain.size(0))
    top5_target.update(prec5[0], input_target_mtrain.size(0))

    # calculate for the source data #######################################################
    if train_loader_source:
        output_source = model_source(input_source_var)
        loss_source = criterion(output_source, target_source_var)
    
        model_source.zero_grad()
        loss_source.backward()
        temp_grad = []
        for param in model_source.parameters():
            temp_grad.append(param.grad.data.clone())
        grad_for_source = temp_grad
    
        prec1, prec5 = accuracy(output_source.data, target_source, topk=(1, 5))
        losses_source.update(loss_source.data[0], input_source.size(0))
        top1_source.update(prec1[0], input_source.size(0))
        top5_source.update(prec5[0], input_source.size(0))

        real_loss = loss_target + loss_source  # for different weight on the loss target
        losses_real.update(real_loss.data[0], input_source.size(0) + input_target_mtrain.size(0))   # here the index for the loss is  input_source.size(0), may be not properly.
    else:
        real_loss = loss_target
        losses_real.update(real_loss.data[0], input_target_mtrain.size(0))

    output_target_temp = model_target(input_target_mtrain_var)
    loss_target_temp = criterion(output_target_temp, target_target_mtrain_var)
    model_target.zero_grad()
    loss_target_temp.backward()
    optimizer.zero_grad()

    if args.first_meta_update:  ## WE Do NOT USE IT.
        count = 0
        for param in model_target.parameters():
            temp_grad = param.grad.data.clone()
            temp_grad.zero_()
            if args.meta_sgd:
                temp_grad = temp_grad + grads_for_task_train[0][count] # just use the gradient with a meta-train batch before the first update
            else:
                temp_grad = temp_grad + grad_for_task_train[count]
            temp_grad = temp_grad
            param.grad.data = temp_grad
            count = count + 1
        optimizer.step()
    
    count = 0
    for param in model_target.parameters():  ## The gradient corresponding to the meta-learning loss
        temp_grad = param.grad.data.clone()
        temp_grad.zero_()
        temp_grad = temp_grad + grad_for_task_mtrain[count]
        temp_grad = temp_grad
        param.grad.data = temp_grad
        count = count + 1
    optimizer.step()
    
    if train_loader_source:  ## The gradient corresponding to the regularization loss
        count = 0
        for param in model_source.parameters():
            temp_grad = param.grad.data.clone()
            temp_grad.zero_()
            temp_grad = temp_grad + grad_for_source[count]
            temp_grad = temp_grad
            param.grad.data = temp_grad
            count = count + 1
        optimizer.step()

    if args.meta_sgd:
        ## update meta train learning rate[to be accomplished]
        meta_train_lr = meta_train_lr_update(args, meta_train_lr, grads_for_task_train, grad_of_meta_test, epoch)

    batch_time.update(time.time() - end)
    if epoch % args.print_freq == 0:
        print('Tr epoch [{0}/{1}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'S@1 {source_top1.val:.3f} ({source_top1.avg:.3f})\t'
              'S@5 {source_top5.val:.3f} ({source_top5.avg:.3f})\t'
              'LS {source_loss.val:.4f} ({source_loss.avg:.4f})\t'
              'T@1 {target_top1.val:.3f} ({target_top1.avg:.3f})\t'
              'T@5 {target_top5.val:.3f} ({target_top5.avg:.3f})\t'
              'LT {target_loss.val:.4f} ({target_loss.avg:.4f})'.format(
               epoch, args.epochs, batch_time=batch_time,
               data_time=data_time, loss=losses_real, source_top1=top1_source, source_top5=top5_source, source_loss=losses_source,
               target_top1=top1_target, target_top5=top5_target, target_loss=losses_target))
    if epoch % args.record_freq == 0:
        log = open(os.path.join(args.log, 'log.txt'), 'a')
        log.write("\n")
        log.write("Tr:epoch: %d, real_loss: %4f, source_T1 acc: %3f, source_T5 acc: %3f, source_loss: %4f, target_T1 acc: %3f, target_T5 acc: %3f, target_loss: %4f"
                  % (epoch, losses_real.avg, top1_source.avg, top5_source.avg, losses_source.avg, top1_target.avg, top5_target.avg, losses_target.avg))
        log.close()
    if args.meta_sgd:
        return train_loader_source_batch, train_loader_target_batch, meta_train_lr
    else:
        return train_loader_source_batch, train_loader_target_batch


def meta_train_update(model_target_temp, meta_train_lr):
    for param, lr_param in zip(model_target_temp.parameters(), meta_train_lr):
        param.data -= lr_param * param.grad.data
    return


def meta_train_lr_update(args, meta_train_lr, grads_for_task_train, grad_of_meta_test, epoch):
    # exp_meta_train_lr = epoch > args.schedule[4] and 3 or epoch > args.schedule[3] and 2 \
    #                     or epoch > args.schedule[2] and 1 or epoch > args.schedule[1] and 0 \
    #                     or epoch > args.schedule[0] and 0 or 0
    # if args.meta_train_lr=0.001, lr_meta_train_lr * grad1 * grad2 = 10^-3 * 10^-2 * 10^-2 = 10^-7 ~ 0
    # equivalently no update meta_train_lr
    exp_meta_train_lr = epoch > args.schedule[4] and 5 or epoch > args.schedule[3] and 4 \
                        or epoch > args.schedule[2] and 3 or epoch > args.schedule[1] and 2 \
                        or epoch > args.schedule[0] and 1 or 0
    lr_meta_train_lr = args.meta_train_lr * (args.gamma ** exp_meta_train_lr)

    count = 0
    for lr, grad1, grad2 in zip(meta_train_lr, grad_of_meta_test, grads_for_task_train[-1]):
        lr -= lr_meta_train_lr * grad1 * (-1) * grad2
        meta_train_lr[count] = lr
        count += 1
    return meta_train_lr


def compute_second_order_grad(args, meta_train_lr, grad_params, grad_for_task_mtrain, model_target_temp):
    # sum = grad_params[0].sum()
    # sum.backward()
    # grad2 = torch.autograd.grad(sum, model_target_temp.parameters(), create_graph=True)

    grad_mul_weight_sum = 0
    for j in range(len(grad_params)):
        grad_param_value = grad_for_task_mtrain[j].clone()
        grad_param_value_var = torch.autograd.Variable(grad_param_value, requires_grad=False)
        grad_mul_weight_sum += (grad_param_value_var * grad_params[j]).sum()
        # grad_mul_weight_sum += grad_params[j].sum()
    grad2_params = torch.autograd.grad(grad_mul_weight_sum, model_target_temp.parameters(), create_graph=False)

    if args.meta_sgd:
        meta_grad_second_order = [grad_for_task_mtrain[i] - meta_train_lr[i] * (e.data.clone()) for i, e in enumerate(grad2_params)]
    else:
        meta_grad_second_order = [grad_for_task_mtrain[i] - meta_train_lr * (e.data.clone()) for i, e in enumerate(grad2_params)]

    return meta_grad_second_order


def validate(val_loader_source, val_loader_target, model_source, model_target, criterion, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_source = AverageMeter()
    top1_source = AverageMeter()
    top5_source = AverageMeter()
    losses_target = AverageMeter()
    # losses_real = AverageMeter()
    top1_target = AverageMeter()
    top5_target = AverageMeter()
    model_source.eval()
    model_target.eval()

    end = time.time()
    if val_loader_source:
        for i, (input_source, target_source) in enumerate(val_loader_source):  # the iterarion in the source dataset.
            data_time.update(time.time() - end)
            target_source = target_source.cuda(async=True)
            input_var = torch.autograd.Variable(input_source, volatile=True)  # volatile is fast in the evaluate model.
            target_var_source = torch.autograd.Variable(target_source, volatile=True)
            output_source = model_source(input_var)
            # calculate for the source data #######################################################
            loss_source = criterion(output_source, target_var_source)
            prec1, prec5 = accuracy(output_source.data, target_source, topk=(1, 5))
            losses_source.update(loss_source.data[0], input_source.size(0))
            top1_source.update(prec1[0], input_source.size(0))
            top5_source.update(prec5[0], input_source.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Te S: [{0}][{1}/{2}]\t'
                      'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'D {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'S@1 {source_top1.val:.3f} ({source_top1.avg:.3f})\t'
                      'S@5 {source_top5.val:.3f} ({source_top5.avg:.3f})\t'
                      'LS {source_loss.val:.4f} ({source_loss.avg:.4f})'.format(
                       epoch, i, len(val_loader_source), batch_time=batch_time,
                       data_time=data_time,  source_top1=top1_source, source_top5=top5_source,
                       source_loss=losses_source,
                       ))

    for i, (input_target, target_target) in enumerate(val_loader_target):  # the iterarion in the source dataset.
        data_time.update(time.time() - end)
        target_target = target_target.cuda(async=True)
        input_var = torch.autograd.Variable(input_target, volatile=True)  # volatile is fast in the evaluate model.
        target_var_target = torch.autograd.Variable(target_target, volatile=True)
        output_target = model_target(input_var)

        # # calculate for the target data#####################################################
        # print(output)
        loss_target = criterion(output_target, target_var_target)
        # #mesure accuracy and record loss
        prec1, prec5 = accuracy(output_target.data, target_target, topk=(1, 5))
        losses_target.update(loss_target.data[0], input_target.size(0))
        top1_target.update(prec1[0], input_target.size(0))
        top5_target.update(prec5[0], input_target.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Te T: [{0}][{1}/{2}]\t'
                  'T {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'D {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'T@1 {target_top1.val:.3f} ({target_top1.avg:.3f})\t'
                  'T@5 {target_top5.val:.3f} ({target_top5.avg:.3f})\t'
                  'LT {target_loss.val:.4f} ({target_loss.avg:.4f})'.format(
                   epoch, i, len(val_loader_target), batch_time=batch_time,
                   data_time=data_time,
                   target_top1=top1_target, target_top5=top5_target, target_loss=losses_target))
    if val_loader_source:
        print(' * Source Dataset (ImageNet) Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1_source, top5=top5_source))
    print(' * Target Dataset ({}) Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(args.dataset, top1=top1_target, top5=top5_target))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n")
    if val_loader_source:
        log.write("                               Test Source:epoch: %d, loss: %4f, Top1 acc: %3f, Top5 acc: %3f" %\
                  (epoch, losses_source.avg, top1_source.avg, top5_source.avg))
        log.write("\n")
    log.write("                               Test Target:epoch: %d, loss: %4f, Top1 acc: %3f, Top5 acc: %3f" %\
              (epoch, losses_target.avg, top1_target.avg, top5_target.avg))
    log.close()
    return top1_target.avg


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


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    exp = epoch > args.schedule[4] and 5 or epoch > args.schedule[3] and 4 or epoch > args.schedule[2] and 3 or epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
    exp_pretrain = epoch > args.schedule[4] and 5 or epoch > args.schedule[3] and 4 or epoch > args.schedule[2] and 3 or epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 2 or 2
    lr = args.lr * (args.gamma ** exp)
    lr_pretrain = args.lr * (args.gamma ** exp_pretrain)
    #print('the lr for new is: ', lr)
    #print('the lr for pretrain is: ', lr_pretrain)
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'pre-trained':
            # param_group['lr'] = lr_pretrain
            param_group['lr'] = 1e-3
        else:
            param_group['lr'] = lr


def adjust_meta_train_learning_rate(optimizer, epoch, args):
    """Adjust the learning rate according the epoch"""
    exp = epoch > args.schedule[4] and 5 or epoch > args.schedule[3] and 4 or epoch > args.schedule[2] and 3 or epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
    meta_train_lr = args.meta_train_lr * (args.gamma ** exp)
    #print('the lr for meta-train is: ', meta_train_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = meta_train_lr
    
    return meta_train_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
