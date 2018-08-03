import argparse


def opts():
    parser = argparse.ArgumentParser(description='Train resnet on the stanford-dogs dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset options
    parser.add_argument('--data-path', type=str, default='/data1/Stanford_Dogs/',
                        help='Root of the target data set (Stanford-Dogs)')
    parser.add_argument('--data-path-source', type=str, default='/data1/ILSVRC2015/',
                        help='Root of the source data set (ImageNet)')
    parser.add_argument('--dataset', type=str, default='stanford_dogs',
                        help='choose target dataset between stanford_dogs/cub200')
    parser.add_argument('--auxiliary-dataset', type=str, default='imagenet',
                        help='choose auxiliary dataset between imagenet/l_bird')
    parser.add_argument('--num-classes-t', type=int, default=120, help='number of classes of target dataset')
    parser.add_argument('--num-classes-s', type=int, default=5000, help='number of classes of source dataset')
    
    # meta-train options
    parser.add_argument('--meta-train-lr', type=float, default=0.001, help='Meta-train learning rate')
    parser.add_argument('--num-updates-for-gradient', type=int, default=1,
                        help='use the gradient of which update iteration')
    parser.add_argument('--meta-sgd', default=False, action='store_true',
                        help='true to use meta-SGD to update parameters')
    parser.add_argument('--second-order-grad', default=False, action='store_true',
                        help='true to use second order gradient to compute the gradient of meta-test')
    parser.add_argument('--first-meta-update', default=False, action='store_true',
                        help='true to use first meta-update gradient to update parameters')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--batch-size-source', '-s', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight-decay', '-wd', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[5000, 7500, 9000],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    # checkpoints
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', type=str, default='', help='Checkpoints path to resume(default none)')
    parser.add_argument('--test-only', '-t', action='store_true', help='Test only flag')

    # Architecture
    parser.add_argument('--arch', type=str, default='resnet34', help='Model name')
    parser.add_argument('--pretrained', action='store_true', help='whether using pretrained model')
    parser.add_argument('--pretrained-checkpoint', type=str, default='', help='choose a start point to train MetaFGNet among pretrained checkpoints')

    # i/o
    parser.add_argument('--log', type=str, default='./checkpoints', help='Log folder')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--test-freq', default=500, type=int,
                        help='print frequency (default: 500)')
    parser.add_argument('--record-freq', default=500, type=int,
                        help='record frequency (default: 500)')
    args = parser.parse_args()
    args.log = args.log + '_' + args.arch + '_' + args.dataset + '_' + str(args.batch_size) + 'Timg_' + args.auxiliary_dataset \
               + '_' + str(args.batch_size_source) + 'Simg_Meta_train_Lr' + str(args.meta_train_lr) + '_' +\
               str(args.num_updates_for_gradient)

    return args
