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

from trainer import train  # For the score computation
from trainer import selectedImages # For the image selection
from selectImage_multiprocess import selected_images_multiprocess
# from trainer import validate  # For the validate (test) process
from opts import opts  # The options for the project
from data.prepare_data import generate_dataloader  # Prepare the data and dataloader
from models.resnet import resnet  # The model construction
import ipdb
best_prec1 = 0

def main():
    global args, best_prec1
    args = opts()
    # args = parser.parse_args()
    model_source, model_target = resnet(args)
   
    # define-multi GPU
    model_source = torch.nn.DataParallel(model_source).cuda()
    model_target = torch.nn.DataParallel(model_target).cuda()
    
    # define loss function(criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # ipdb.set_trace()
    if args.resume:
        if os.path.isfile(args.resume):
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
#            # args.start_epoch = checkpoint['epoch']
#            # best_prec1 = checkpoint['best_prec1']
            model_source.load_state_dict(checkpoint['source_state_dict'])
            model_target.load_state_dict(checkpoint['target_state_dict'])
           
            print("==> loaded checkpoint '{}'(epoch {})"
                 .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from is not existed', args.resume)
#    # else:
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    log = open(os.path.join(args.log, 'log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()

    cudnn.benchmark = True
    # process the data and prepare the dataloaders.
    # ipdb.set_trace()
    train_loader = generate_dataloader(args)
    # train(train_loader, model_source, model_target, 1, args)
    selectedImages(args)
    selected_images_multiprocess(args)




def save_checkpoint(state, is_best, args):
    filename = 'checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(state, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()





