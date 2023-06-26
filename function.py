from time import localtime, strftime

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import shutil
import random
import torch
import kornia.augmentation as A


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, max_up=False):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.max_up = max_up

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn)
        if self.transform is not None:
            img = self.transform(img)

        if self.max_up:
            label_onehot = torch.zeros(10, dtype=torch.float32)
            label_onehot[label] = 1.
            return img, label_onehot
        else:
            return img, label

    def __len__(self):
        return len(self.imgs)

    def __getname__(self, index):
        fn, label = self.imgs[index]
        img_name = fn.split('/')[-1]
        return img_name


def setDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        print('Dir removed')
        os.mkdir(filepath)


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    def __init__(self):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((32, 32), padding=5), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(10), p=0.5)
        self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)
        self.random_erasing = A.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def folder_name(args):
    argument_list = []
    if args.smooth:
        argument_list.append('smooth')
    if args.B_C:
        argument_list.append('B&C')
    if args.blur:
        argument_list.append('blur')
    if args.warping:
        argument_list.append('warping')
    if args.sinusoidal:
        argument_list.append('sinusoidal')
    if args.patch:
        argument_list.append('patch')
    if args.adaptive:
        argument_list.append('adaptive')
    if args.random:
        argument_list.append('random')

    argument_list.append('da=' + args.da)
    argument_list.append('ir=' + str(args.ir))
    argument_list.append('lam=' + str(args.lam))
    date_time = strftime('%Y-%m-%d_%H:%M:%S', localtime())
    argument_list.append('time=' + date_time)

    res_folder_name = ''
    for j in range(len(argument_list)):
        res_folder_name += '' + argument_list[j] + ''
        if j < len(argument_list) - 1:
            res_folder_name += '--'

    return res_folder_name


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


def setup_seed(seed=3407):
    os.environ['PYTHONASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

    torch.backends.cudnn.benchmark = True
