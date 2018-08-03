import argparse


def opts():
    parser = argparse.ArgumentParser(description='Download score of auxiliary data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset options
    parser.add_argument('--data-path', type=str, default='/data1/ILSVRC2015/',
                        help='Root of the data set')
    parser.add_argument('--dataset', type=str, choices=['stanford_dogs', 'cub200'],
                        help='choose between stanford_dogs/cub200')
    parser.add_argument('--auxiliary-dataset', type=str, default='imagenet',
                        help='choose auxiliary dataset between imagenet/l_bird')
    parser.add_argument('--num-classes-t', type=int, default=120, help='number of classes of target dataset')
    parser.add_argument('--num-classes-s', type=int, default=1000, help='number of classes of source dataset')
    parser.add_argument('--score-path', default='/data1/ILSVRC2015/images_scores/', type=str, help='path to save images score')
    parser.add_argument('--selected-image-path', default='/data1/ILSVRC2015/images_selected/', type=str, help='path to save selected image')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight-decay', '-wd', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # checkpoints
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', type=str, default='', help='Checkpoints path to resume(default none)')
    parser.add_argument('--test-only', '-t', action='store_true', help='Test only flag')
    # Architecture
    parser.add_argument('--arch', type=str, default='resnet34', help='Model name')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='whether using pretrained model')
    # i/o
    parser.add_argument('--log', type=str, default='./checkpoints', help='Log folder')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
                        
    # Edgebox_score_selection
    parser.add_argument('--criterion', default='topk', type=str,
                         help='it can be chosen between topk & ratio_threshold & score_threshold')
    parser.add_argument('--select-ratio', default=0.05, type=float, help='the select ratio used in ratio threshold module')
    parser.add_argument('--select-score', default=200, type=float, help='the select score used in score threshold module')
    parser.add_argument('--topk', default=40, type=int, help='the topk selected proposals in the topk module')
    args = parser.parse_args()
    return args
