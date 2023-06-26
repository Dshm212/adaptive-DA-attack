import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    # Datasets
    parser.add_argument("--clean_root", type=str, default="/mnt/ssd4/chaohui/datasets/CIFAR10/")
    parser.add_argument('-d', '--dataset', default='CIFAR10', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--ir', default=0.01, type=float, help='injection ratio')
    parser.add_argument('--ext', default='.png', type=str, help='file extension')
    # Backdoor parameter
    parser.add_argument('--model', default='ResNet', type=str)

    parser.add_argument('--trigger_type', default='DA', type=str, help='trigger type: DA, CLBA, Sig, ReFool, HTBA')

    parser.add_argument('--delta', default=20, type=float)
    parser.add_argument('--kernel-size', default=3, type=int)
    parser.add_argument('--brightness-limit', default=0.3, type=float)
    parser.add_argument('--sigma', default=0.75, type=float)
    parser.add_argument('--lam', default=0.5, type=float)
    parser.add_argument('--c1', default=1.0, type=float)
    parser.add_argument('--c2', default=1.0, type=float)
    parser.add_argument('--smooth', default=True, type=bool)
    parser.add_argument('--B_C', default=True, type=bool)
    parser.add_argument('--blur', default=True, type=bool)
    parser.add_argument('--ablation_channel', default='0', type=int, help='the channel number for ablation study')

    parser.add_argument('--height', default=32, type=int)
    parser.add_argument('--width', default=32, type=int)
    parser.add_argument('--bs', default=32, type=int)

    parser.add_argument('--target-label', default=5, type=int)
    # Results
    parser.add_argument("--res_root", type=str, default="/mnt/ssd4/chaohui/results/CLBA/CIFAR10/")
    parser.add_argument("--res_sub", type=str)
    # Optimization options
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    # parser.add_argument('--epochs', default=80, type=int, metavar='N',
    #                     help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[40, 80],
                        help='Decrease learning rate at these epochs.')
    # parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60],
    #                     help='Decrease learning rate at these epochs.')
    parser.add_argument('--seed_list', type=list, nargs='+', default=[826, 212, 1038, 614, 1107])
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--da', '--data-augmentation', default='none', type=str,
                        help='data augmentation mode: none, mix-up, cut-mix, max-up (defalt: none)')
    parser.add_argument('--max-up-num', default=2, type=int)
    parser.add_argument('--ct', default=False, type=bool)
    parser.add_argument('--adaptive', default=False, type=bool)
    parser.add_argument('--random', default=False, type=bool)
    parser.add_argument('--repeats', default=3, type=int)
    # Device options
    parser.add_argument('--gpu-id', default='1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    return parser