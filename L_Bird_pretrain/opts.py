import argparse


def opts():
    parser = argparse.ArgumentParser(description='Train resnet on the cub dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='/home/lab-zhangyabin/project/fine-grained/CUB_200_2011/',
                        help='Root of the data set')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='the pretrained modelp')
    parser.add_argument('--dataset', type=str, choices=['l-bird', 'cub200'],
                        help='choose between l-bird/cub200')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=161, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # checkpoints
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', type=str, default='', help='Checkpoints path to resume(default none)')
    parser.add_argument('--test_only', '-t', action='store_true', help='Test only flag')
    # Architecture
    parser.add_argument('--arch', type=str, default='resnet34', help='Model name')
    parser.add_argument('--pretrain', action='store_true', help='whether using pretrained model')
    parser.add_argument('--newfc', action='store_true', help='whether initialize the classifier')
    parser.add_argument('--numclass_old', type=int, default=1000, help='class Number of the pretrained model')
    parser.add_argument('--numclass_new', type=int, default=10320, help='class Number of new model to be trained or fine-tuned')
    # i/o
    parser.add_argument('--log', type=str, default='./checkpoints', help='Log folder')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()

    return args
