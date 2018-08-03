import argparse


def opts():
    parser = argparse.ArgumentParser(description='Train resnet on the cub dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', type=str, default='/data1/Stanford_Dogs/',
                        help='Root of the data set')
    parser.add_argument('--dataset', type=str, default='stanford_dogs',
                        help='choose between stanford_dogs/cub200')
    parser.add_argument('--num-classes-t', type=int, default=120, help='number of classes of target dataset')
    parser.add_argument('--num-classes-s', type=int, default=1000, help='number of classes of source dataset')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=160, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight-decay', '-wd', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[79, 119],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # checkpoints
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', type=str, default='', help='Checkpoints path to resume(default none)')
    parser.add_argument('--test-only', '-t', action='store_true', help='Test only flag')
    parser.add_argument('--num-train-iter', type=int, default=-1, help='number of trainig iterations')
    # Architecture
    parser.add_argument('--arch', type=str, default='resnet34', help='Model name')
    parser.add_argument('--pretrained', action='store_true', help='whether using pretrained model')
    parser.add_argument('--new-fc', action='store_true', help='whether initializing fc layer')
    # i/o
    parser.add_argument('--log', type=str, default='./checkpoints', help='Log folder')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    args.log = args.log + '_' + str(args.dataset) + '_lr_' + str(args.lr) + '_momentum_' + str(args.momentum) + '_wd_' + str(args.weight_decay) + '_epochs_' + str(args.epochs)
    return args
