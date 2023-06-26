import copy
import random

import numpy as np
import torchattacks
import torchvision
from torch import optim, nn
from torch.cuda.amp import GradScaler

import config
import image_processing
import torchvision.transforms as T
import pyiqa
import torch
import os

from function import MyDataset

args = config.get_arguments().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

adv_path = '/mnt/ssd4/chaohui/backdoor_projects/CLBA/CIFAR10/adv_classifier/AlexNet/checkpoint/adv_classifier.pth'
adv_model = torch.load(adv_path)
adv_model.cuda().eval()

# adv_path = '/mnt/ssd4/chaohui/results/CLBA/CIFAR10/IR/2022-12-29_14:59/run_0/checkpoint/checkpoint.pth'
# adv_model = torch.load(adv_path)
# adv_model.cuda().eval()

# adv_model = torchvision.models.resnet18(pretrained=True, progress=False).cuda().eval()

lpips_metric = pyiqa.create_metric('lpips', device=torch.device('cuda'), as_loss=True)
psnr_metric = pyiqa.create_metric('psnr', device=torch.device('cuda'), as_loss=True)
gmsd_metric = pyiqa.create_metric('gmsd', device=torch.device('cuda'), as_loss=True)

normalize = T.Normalize(
    mean=(0.491, 0.482, 0.446),
    std=(0.247, 0.243, 0.261)
)

preprocessing = T.ToTensor()


def image_poisoning(ori, train, args):
    if args.trigger_type == "DA":
        res = DA_backdoor(ori, train, args)
    elif args.trigger_type == "Sig":
        res = Sig_backdoor(ori, train, args)
    elif args.trigger_type == "CLBA":
        res = CLBA_backdoor(ori, train, args)
    elif args.trigger_type == "ReFool":
        res = ReFool_backdoor(ori, train, args)
    elif args.trigger_type == "HTBA":
        res = HTBA_backdoor(ori, train, args)
    else:
        raise ValueError("Invalid backdoor type: ", args.trigger_type)

    res = np.array(res, dtype='uint8')
    return res


def CLBA_backdoor(ori, train, args):
    def CLBA_adv_pert(img):
        img_tensor = preprocessing(img).unsqueeze(0).type(torch.FloatTensor).cuda()
        adv_label = torch.argmax(adv_model(normalize(img_tensor)))
        adv_label = torch.reshape(adv_label, (1,))
        ori_tensor = copy.deepcopy(img_tensor)
        atk = torchattacks.PGD(adv_model, eps=16 / 255, alpha=2 / 255, steps=20)
        img_tensor = atk(ori_tensor, adv_label)
        adv_image = img_tensor.squeeze(0).detach().cpu().numpy()
        adv_image = np.transpose(adv_image, (1, 2, 0)) * 255
        adv_image = np.array(adv_image, dtype="uint8")
        return adv_image

    p_list = []
    for i in range(len(ori)):
        # if train:
        #     perturbed = CLBA_adv_pert(ori[i])
        # else:
        #     perturbed = ori[i]
        perturbed = CLBA_adv_pert(ori[i])
        poisoned = image_processing.patch(perturbed)
        p_list.append(poisoned)
    p_array = np.array(p_list)

    return p_array


def Sig_backdoor(ori, train, args):
    p_list = []
    for i in range(len(ori)):
        poisoned = image_processing.sinusoidal(ori[i], args.delta)
        p_list.append(poisoned)
    p_array = np.array(p_list)

    return p_array


def ReFool_backdoor(ori, train, args):
    p_list = []
    for i in range(len(ori)):
        poisoned = image_processing.reflection(ori[i], args.lam)
        p_list.append(poisoned)
    p_array = np.array(p_list)

    return p_array


def HTBA_backdoor(ori, train, args):
    def HTBA_pert(ori, patched):
        adv_tensor = preprocessing(ori).unsqueeze(0).type(torch.FloatTensor).cuda()
        patched_tensor = preprocessing(patched).unsqueeze(0).type(torch.FloatTensor).cuda()
        activation = {}

        ori_tensor = adv_tensor.detach().clone()

        alpha = 2 / 255
        eps = 40 / 255

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output

            return hook

        adv_model.layer4[1].shortcut.register_forward_hook(get_activation('layer'))

        for step in range(30):
            adv_tensor.requires_grad = True

            adv_feats = adv_model(normalize(adv_tensor))
            feats_adv = activation['layer']
            patched_feats = adv_model(normalize(patched_tensor))
            feats_patched = activation['layer']
            loss = ((feats_adv - feats_patched) ** 2).sum()
            print(loss)

            loss.backward()

            tmp_tensor = adv_tensor - alpha * adv_tensor.grad.sign()
            eta = torch.clamp(tmp_tensor - ori_tensor, min=-eps, max=eps)
            adv_tensor = torch.clamp(ori_tensor + eta, min=0, max=1).detach_()

        adv_image = adv_tensor.squeeze(0).detach().cpu().numpy()
        adv_image = np.transpose(adv_image, (1, 2, 0)) * 255
        adv_image = np.array(adv_image, dtype="uint8")

        return adv_image

    clean_train_set = MyDataset(txt=args.clean_root + '/clean_train.txt', transform=None)
    num_samples = len(clean_train_set)
    p_list = []

    for i in range(len(ori)):
        # if train:
        #     base_label = args.target_label
        #     while base_label != 0:
        #         num = random.randint(0, num_samples-1)
        #         base_img, base_label = clean_train_set.__getitem__(num)
        #
        #     base_img = np.array(base_img)
        #     patched = image_processing.random_patch(base_img)
        #     poisoned = HTBA_pert(ori[i], patched)
        #     # poisoned = patched
        # else:
        #     poisoned = image_processing.random_patch(ori[i])

        base_label = args.target_label
        while base_label != 0:
            num = random.randint(0, num_samples - 1)
            base_img, base_label = clean_train_set.__getitem__(num)

        base_img = np.array(base_img)
        patched = image_processing.random_patch(base_img)
        poisoned = HTBA_pert(ori[i], patched)

        p_list.append(poisoned)
    p_array = np.array(p_list)

    return p_array


def DA_backdoor(ori, train, args):
    def DA_embedding(img, args):
        # if args.smooth:
        #     img = image_processing.Smooth(img, args.kernel_size)
        # if args.B_C:
        #     img = image_processing.BrightnessContrast(img, args.brightness_limit)
        # if args.blur:
        #     img = image_processing.Blur(img, args.sigma)

        if args.ablation_channel == 0:
            img = image_processing.CH0(img)
            img = image_processing.BrightnessContrast(img, args.brightness_limit)
            img = image_processing.Smooth(img, args.kernel_size)
        elif args.ablation_channel == 1:
            img = image_processing.Blur(img, args.sigma)
            img = image_processing.CH1(img)
            img = image_processing.Smooth(img, args.kernel_size)
        elif args.ablation_channel == 2:
            img = image_processing.Blur(img, args.sigma)
            img = image_processing.BrightnessContrast(img, args.brightness_limit)
            img = image_processing.CH2(img)
        elif args.ablation_channel == -1:
            img = image_processing.CH0(img)
            img = image_processing.CH1(img)
            img = image_processing.CH2(img)
        else:
            raise ValueError("Invalid model ablation channel number: ", args.ablation_channel)

        return img

    c_list = []
    p_list = []
    for i in range(len(ori)):
        c_list.append(ori[i])
        p_list.append(DA_embedding(ori[i], args))
    c_array = np.array(c_list)
    p_array = np.array(p_list)

    if train:
        if args.adaptive:
            res = adaptive_interpolation(c_array, p_array, args)
        elif args.random:
            bs = c_array.shape[0]
            mask = np.random.uniform(low=0.5, high=0.9, size=(bs, 1, 1, 1))
            res = mask * p_array + (1 - mask) * c_array
        else:
            res = p_array
    else:
        res = p_array

    return res


def adaptive_interpolation(c_array, p_array, args):
    bs = c_array.shape[0]
    c_tensor = torch.stack([preprocessing(c_array[i]) for i in range(bs)])
    p_tensor = torch.stack([preprocessing(p_array[i]) for i in range(bs)])

    c_tensor = c_tensor.type(torch.FloatTensor).cuda()
    p_tensor = p_tensor.type(torch.FloatTensor).cuda()

    mask = torch.ones((bs, 1, 32, 32)) * random.uniform(0, 0.4) + 0.5
    mask = mask.cuda().requires_grad_(True)

    optimizer = torch.optim.Adam([{"params": mask}], lr=0.01)
    scaler = GradScaler()
    m = nn.ReLU()
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook

    adv_model._modules['features'][11].register_forward_hook(get_activation('layer'))

    for i in range(200):
        mask.requires_grad = True
        res_tensor = p_tensor * mask.expand_as(c_tensor) + c_tensor * (1 - mask).expand_as(c_tensor)

        lpips_score = lpips_metric(res_tensor, c_tensor)
        psnr_score = psnr_metric(res_tensor, c_tensor)
        gmsd_score = gmsd_metric(res_tensor, c_tensor)

        res_feats = adv_model(normalize(res_tensor))
        feats_res = activation['layer']
        poison_feats = adv_model(normalize(p_tensor))
        feats_poison = activation['layer']

        feats_loss = torch.norm(feats_res - feats_poison) / torch.norm(feats_poison)
        adv_loss = feats_loss

        lpips_loss = torch.mean(m(lpips_score - 0.025))
        psnr_loss = torch.mean(m(29 - psnr_score))
        gmsd_loss = torch.mean(m(gmsd_score - 0.01))

        loss = args.c1 * adv_loss + args.c2 * (lpips_loss + psnr_loss + gmsd_loss)

        print("======Loss info======")
        print("binary loss =  {:.4f}".format(adv_loss))
        print("lpips loss =  {:.4f}, psnr loss =  {:.4f}, gmsd loss =  {:.4f}".format(lpips_loss, psnr_loss, gmsd_loss))
        print("=======ratio info======")
        print("mask mean = {:.4f}".format(torch.mean(mask).item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            torch.clip_(mask, 0.5, 0.9)

    res = p_tensor * mask.expand_as(c_tensor) + c_tensor * (1 - mask).expand_as(c_tensor)

    res = res.detach().cpu().numpy()
    res = np.transpose(res, (0, 2, 3, 1)) * 255
    res = np.array(res, dtype="uint8")
    return res
