import random
import torch
import numpy as np

from torch.utils.data import Dataset
from torch.nn.modules.module import Module


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.dirichlet([alpha, alpha])[0]
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.dirichlet([alpha, alpha])[0]
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = x
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


class CutMix(Dataset):
    def __init__(self, dataset, m=1, alpha=1.0):
        self.dataset = dataset
        self.m = m
        self.alpha = alpha

    def __getitem__(self, index):
        m_imgs = []
        m_lbls = []
        for _ in range(self.m):
            img, lbl1 = self.dataset[index]

            # generate mixed sample
            lam = np.random.dirichlet([self.alpha, self.alpha])[0]
            rand_index = random.choice(range(len(self)))

            img2, lbl2 = self.dataset[rand_index]

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lbl = lbl1 * lam + lbl2 * (1. - lam)
            m_imgs.append(img)
            m_lbls.append(lbl)
        m_imgs = torch.stack(m_imgs, 0)
        m_lbls = torch.stack(m_lbls, 0)
        return m_imgs, m_lbls

    def __len__(self):
        return len(self.dataset)


class MaxupCrossEntropyLoss(Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input, target):
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        batch_size = target.shape[0]
        target = target.reshape(input.shape)

        loss = torch.sum(-target * logsoftmax(input), dim=1)
        loss, _ = loss.reshape((batch_size, self.m)).max(1)
        loss = torch.mean(loss)
        return loss
