from __future__ import print_function
import glob
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import yaml
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from function import MyDataset, PostTensorTransform, setDir, create_exp_dir, save_checkpoint, setup_seed
import DA_defense
import config
from backdoor_train_dataset import backdoor_train_image
from backdoor_test_dataset import backdoor_test_image
from utils import Logger, AverageMeter, accuracy
from classifier_models import ResNet18, VGG, EfficientNetB0, DenseNet121

args = config.get_arguments().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


def main(args):
    # Use CUDA
    use_cuda = torch.cuda.is_available()

    global best_acc
    best_acc = 0  # best test accuracy

    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    date_time = time.strftime('%Y-%m-%d_%H:%M', time.localtime())
    args.res_sub = os.path.join(args.res_root, args.res_sub)
    args.res_folder = os.path.join(args.res_sub, date_time)
    args.data = os.path.join(args.res_folder, "data")

    setDir(args.res_folder)
    setDir(args.data)

    create_exp_dir(os.path.join(args.res_folder), scripts_to_save=glob.glob('*.py'))
    # Data
    print('==> Preparing dataset %s' % args.dataset)

    backdoor_train_image(args)
    backdoor_test_image(args)

    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )

    transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])

    for repeats in range(args.repeats):
        setup_seed(seed=args.seed_list[repeats])

        args.run = os.path.join(args.res_folder, "run_" + str(repeats))
        args.log = os.path.join(args.run, "log")
        args.checkpoint = os.path.join(args.run, "checkpoint")
        setDir(args.run)
        setDir(args.log)
        setDir(args.checkpoint)

        with open(os.path.join(args.run, "config.yml"), "w") as f:
            yaml.dump(args, f, sort_keys=False)

        if args.da != 'max-up':
            trainloader = torch.utils.data.DataLoader(
                MyDataset(txt=args.data + '/backdoor_train.txt', transform=transform),
                batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        else:
            trainloader = torch.utils.data.DataLoader(
                DA_defense.CutMix(MyDataset(txt=args.data + '/backdoor_train.txt', transform=transform, max_up=True),
                                  args.max_up_num),
                batch_size=args.train_batch, shuffle=True, num_workers=args.workers
            )

        testloader = torch.utils.data.DataLoader(
            MyDataset(txt=args.clean_root + '/clean_test.txt', transform=transform),
            batch_size=args.test_batch, shuffle=True, num_workers=args.workers)
        backdoor_loader = torch.utils.data.DataLoader(
            MyDataset(txt=args.data + '/backdoor_test.txt', transform=transform),
            batch_size=args.test_batch, shuffle=True, num_workers=args.workers)

        if args.model == "EfficientNet":
            model = EfficientNetB0()
            model.cuda()

        elif args.model == "DenseNet":
            model = DenseNet121()
            model.cuda()

        elif args.model == "VGG":
            model = VGG("VGG16")
            model.cuda()

        elif args.model == "ResNet":
            model = ResNet18(num_classes=10)
            model.cuda()

            # adv_path = '/mnt/ssd4/chaohui/backdoor_projects/CLBA/CIFAR10/adv_classifier/ResNet/adv_classifier.pth'
            # model = torch.load(adv_path)
            #
            # for param in model.parameters():
            #     param.requires_grad = False
            #     pass
            # model.linear = nn.Linear(model.linear.in_features, 10)
            # model.cuda()

        else:
            raise ValueError("Invalid model architecture: ", args.model)

        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.schedule, args.gamma)
        scaler = GradScaler()

        tf_writer = SummaryWriter(log_dir=args.log)
        title = 'CIFAR10'
        logger = Logger(os.path.join(args.log, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Success Rate.'])

        print("Data augmentation mode: ", args.da)

        # Train and val
        for epoch in range(start_epoch, args.epochs):
            lr_tmp = scheduler.get_last_lr()[0]
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr_tmp))

            start = time.time()

            train_loss, train_acc = train(args, trainloader, model, criterion, optimizer, scaler, use_cuda)
            val_loss, val_acc = test(testloader, model, criterion, use_cuda)
            backdoor_loss, success_rate = test(backdoor_loader, model, criterion, use_cuda)

            end = time.time()
            print('time:\t', np.round(end - start, 2))
            print('train loss:\t', train_loss)
            print('validation loss:\t', val_loss)
            print('train accuracy:\t', train_acc)
            print('validation accuracy:\t', val_acc)
            print('success rate:\t', success_rate)

            if not epoch % 1:
                tf_writer.add_scalars(
                    "Clean Accuracy", {"train_acc": train_acc, "val_acc": val_acc}, epoch
                )
                tf_writer.add_scalars(
                    "Clean loss", {"train_loss": train_loss, "val_loss": val_loss}, epoch
                )
                tf_writer.add_scalars(
                    "Attack success rate", {"success rate": success_rate, "val_acc": val_acc}, epoch
                )

            # append logger file
            logger.append([lr_tmp, train_loss, val_loss, train_acc, val_acc, success_rate])

            # save model
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_checkpoint(model, is_best, checkpoint=args.checkpoint)

            scheduler.step()

        logger.close()

        print('Best acc:')
        print(best_acc)


def train(args, trainloader, model, criterion, optimizer, scaler, use_cuda):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    trans = PostTensorTransform().cuda()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if args.da == "none":
            inputs = trans(inputs)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        elif args.da == "mix-up":
            inputs = trans(inputs)
            inputs, targets_a, targets_b, lam = DA_defense.mixup_data(x=inputs, y=targets, alpha=1.0, use_cuda=True)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
            with autocast():
                outputs = model(inputs)
                loss = DA_defense.mix_criterion(criterion, outputs, targets_a, targets_b, lam)
        elif args.da == "cut-mix":
            inputs = trans(inputs)
            inputs, targets_a, targets_b, lam = DA_defense.cutmix_data(x=inputs, y=targets, alpha=1.0, use_cuda=True)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
            with autocast():
                outputs = model(inputs)
                loss = DA_defense.mix_criterion(criterion, outputs, targets_a, targets_b, lam)
        elif args.da == "max-up":
            criterion = DA_defense.MaxupCrossEntropyLoss(args.max_up_num)
            inputs = inputs.reshape((inputs.shape[0] * args.max_up_num, 3, args.height, args.width))
            inputs = trans(inputs)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            raise ValueError("Invalid data augmentation mode: ", args.da)

        # measure accuracy and record loss
        if args.da != 'max-up':
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))

        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # loss.backward()
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()

    return losses.avg, top1.avg


def test(testloader, model, criterion, use_cuda):
    global best_acc

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

    return losses.avg, top1.avg


if __name__ == "__main__":
    main(args=args)
