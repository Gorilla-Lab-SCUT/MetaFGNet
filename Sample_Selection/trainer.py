import time
import torch
import os
import torch.nn as nn
# from data.prepare_data_for_path import generate_dataset_new
import ipdb


def train(train_loader, model_source, model_target, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1_s = AverageMeter()
    top5_s = AverageMeter()
    model_source.eval()
    model_target.eval()
    end = time.time()
    softmax = nn.Softmax()
    # dict_to_download = {}
    # ipdb.set_trace()
    # train_dataset = generate_dataset_new(args)
    # path_to_download = [train_dataset.__getitem__(i)[0] for i in range(len(train_dataset))]
    # score_sm_to_download = []
    # score_tm_to_download = []
    # score_sum_norm_negativeSet0 = []
    # score_sum_noNorm_noNegativeSet0 = []
    score_sm_tm = torch.zeros(args.num_classes_s + args.num_classes_t)
    dict_to_download = {}
    count = 0
    for i, batch in enumerate(train_loader):
        batch_paths = batch[0]
        input = batch[1]
        target = batch[2]
        target = target.cuda(async=True)
        # ipdb.set_trace() first batch target: all 0
        input_var = torch.autograd.Variable(input, volatile=True)
        # compute output
        output_s_score = model_source(input_var)
        output_t_score = model_target(input_var)
        output_s_prob = softmax(output_s_score)
        data_time.update(time.time() - end)
        # measure accuracy and record loss
        prec1_s, prec5_s = accuracy(output_s_prob.data, target, topk=(1, 5))
        top1_s.update(prec1_s[0], input.size(0))
        top5_s.update(prec5_s[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        for j in range(input.size(0)):
            # a = output.data.cpu()
            # print(a[j][target[j]])
            # print(target[j])
            score_sm_tm[:args.num_classes_s] = output_s_score.data.cpu()[j].clone()
            # Because output_s_score.data.cpu()[j] is a element of the tensor output_s_score.data.cpu(),
            # a element of it occupys the same size of  memory as the whole tensor! If we save a element 
            # of a tensor to a file(disk), it will take up the same memory space as the tensor. 
            # 128*1000*1200*4/1024/1024 * 128 = 74G
            score_sm_tm[args.num_classes_s:] = output_t_score.data.cpu()[j].clone()
            # score_sum_noNorm_noNegativeSet0.append(torch.sum(score_sm_tm[args.num_classes_s:]))
            score_sm_tm[score_sm_tm < 0] = 0
            score_sm_tm /= torch.norm(score_sm_tm) # comment for summing up target scores without norm
            score_sum = torch.sum(score_sm_tm[args.num_classes_s:])
            # score_sum_norm_negativeSet0.append(torch.sum(score_sm_tm[args.num_classes_s:]))
            image_path = batch_paths[j]
            if args.auxiliary_dataset == 'imagenet':
                image_score_path = image_path.replace(args.data_path, args.score_path).replace('.JPEG', '.pth.tar')
            elif args.auxiliary_dataset == 'l_bird':
                image_score_path = image_path.replace(args.data_path, args.score_path).replace('.jpg', '.pth.tar')
            else:
                raise ValueError('Unavailable auxiliary dataset!')
            pos = image_score_path.rfind('/')
            score_dir = image_score_path[:pos]
            if not os.path.exists(score_dir):
                os.makedirs(score_dir)
            dict_to_download['score_sum'] = score_sum
            if not os.path.isfile(image_score_path):
                torch.save(dict_to_download, image_score_path)

        # label_to_download.append(target)
        if i % args.print_freq == 0:
            print('Test: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1_Sm {top1_s.val:.3f} ({top1_s.avg:.3f})\t'
                  'Prec@5_Sm {top5_s.val:.3f} ({top5_s.avg:.3f})'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   top1_s=top1_s, top5_s=top5_s))

    print(' * Source model Prec@1 {top1_s.avg:.3f} Prec@5 {top5_s.avg:.3f}'
          .format(top1_s=top1_s, top5_s=top5_s))

    # dict_to_download['path'] = path_to_download  # Path of images save in the last file('...10.pth.tar')
    # dict_to_download['score_sm'] = score_sm_to_download
    # dict_to_download['score_tm'] = score_tm_to_download
    # dict_to_download['label'] = label_to_download
    # dict_to_download['score_sum_noNorm_noNegativeSet0'] = score_sum_noNorm_noNegativeSet0
    # dict_to_download['score_sum_norm_negativeSet0'] = score_sum_norm_negativeSet0

    # filename = args.source_image_file_name + '_score_sum.pth.tar'
    # dir_save_file = os.path.join(args.log, filename)
    # torch.save(dict_to_download, dir_save_file)


def selectedImages(args):
    criterion = args.criterion  # options: ratio_threshold, score_threshold, topk
    select_ratio = args.select_ratio
    select_score = args.select_score
    topk = args.topk
    
    score_dir_list = []
    if args.auxiliary_dataset == 'imagenet':
        score_path = os.path.join(args.score_path, 'Data/CLS-LOC/train')
    elif args.auxiliary_dataset == 'l_bird':
        score_path = args.score_path
    else:
        raise ValueError('Unavailable auxiliary dataset!')
    classes_name = os.listdir(score_path)
    classes_name.sort()
    for class_name in classes_name:
        class_dir = os.path.join(score_path, class_name)
        exps_name = os.listdir(class_dir)
        score_dir_list.extend([os.path.join(class_dir, exp_name) for exp_name in exps_name])
    
    scores = torch.FloatTensor(len(score_dir_list)).zero_()
    # ipdb.set_trace()
    for i in range(len(score_dir_list)):
        if i % 10 == 0:
            print('Have computated: ', i)
        scores[i] = torch.load(score_dir_list[i])['score_sum']
        # scores[i] = torch.sum(torch.load(score_dir_list[i])['score_sum'][args.num_classes_s:])
    
    score_descending, ind = torch.sort(scores, descending=True)
    if criterion == 'score_threshold':
        selected_ind = ind[score_descending > select_score]
    elif criterion == 'ratio_threshold':
        threshold_score = score_descending[round(score_descending.shape[0] * select_ratio)]
        selected_ind = ind[score_descending > threshold_score] # indexes of the selected images
        # selected_ind = ind[score_descending < threshold_score] # indexes of the abandoned images
    elif criterion == 'topk':
        if topk > score_descending.shape[0]:
            topk = score_descending.shape[0]
        # selected_ind = ind[:topk]
        threshold_score = score_descending[topk]
        selected_ind = ind[score_descending > threshold_score]
    else:
        raise ValueError('not defined criterion')
    torch.save(score_dir_list, os.path.join(args.log, 'score_dir_list.pth.tar'))
    torch.save(selected_ind, os.path.join(args.log, 'selected_ind.pth.tar'))

    ############################################ the below is a single process to select the related source images to generate a refined source dataset.#########
    # for j in range(selected_ind.shape[0]):
    #     if args.auxiliary_dataset == 'imagenet':
    #         original_image_dir = score_dir_list[selected_ind[j]].replace(args.score_path, args.data_path).replace('.pth.tar', '.JPEG')
    #     elif args.auxiliary_dataset == 'l_bird':
    #         original_image_dir = score_dir_list[selected_ind[j]].replace(args.score_path, args.data_path).replace('.pth.tar', '.jpg')
    #     else:
    #         raise ValueError('Unavailable auxiliary dataset!')
    #     #selected_image_dir = original_image_dir.replace(args.data_path, args.selected_image_path)
    #     #pos = selected_image_dir.rfind('/')
    #     #class_dir = selected_image_dir[:pos]
    #     #if not os.path.exists(class_dir):
    #     #    os.makedirs(class_dir)
    #     # os.system('cp ' + original_image_dir + ' ' + selected_image_dir)
    #     if os.path.exists(original_image_dir):
    #         if j % 1000 == 0:
    #             start = time.time()
    #         os.system('rm ' + original_image_dir)
    #     #os.system('mv ' + original_image_dir + ' ' + selected_image_dir) # move the abandoned images to another folder
    #         if j % 1000 == 0:
    #             print(str(time.time() - start))


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    log = open(os.path.join(args.log, 'log.txt'), 'a')
    log.write("\n")
    log.write("                               Train:epoch: %d, loss: %4f, Top1 acc: %3f, Top5 acc: %3f" %\
              (epoch, losses.avg, top1.avg, top5.avg))
    log.close()
    return top1.avg


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
    exp = epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
    lr = args.lr * (args.gamma ** exp)
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'pre-trained':
            param_group['lr'] = 1e-3
        else:
            param_group['lr'] = lr


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
