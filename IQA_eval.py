import torchvision.transforms as transforms
from PIL import Image
import ssl
import os
import pyiqa
import torch
import statistics
import numpy as np
from matplotlib import pyplot as plt

from function import MyDataset

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
use_cuda = torch.cuda.is_available()
ssl._create_default_https_context = ssl._create_unverified_context

mode = "train"
device = "cuda"

transform_test = transforms.Compose([
    transforms.ToTensor()
])

root = "/mnt/ssd4/chaohui/results/CLBA/CIFAR10/transformation/2023-06-23_17:00/data"

lpips_list = []
psnr_list = []
gmsd_list = []
ssim_list = []
niqe_clean_list = []
niqe_poison_list = []
count = 0

if mode == "train":
    clean_txt = os.path.join(root, "train_instance.txt")
    poison_txt = os.path.join(root, "train_poison.txt")
elif mode == "test":
    clean_txt = os.path.join(root, "test_instance.txt")
    poison_txt = os.path.join(root, "backdoor_test.txt")
else:
    raise ValueError("Invalid mode: ", mode)

clean_data = MyDataset(txt=clean_txt, transform=None)
poison_data = MyDataset(txt=poison_txt, transform=None)

num_of_samples = len(clean_data)
print("Number of samples to compute GMSD: ", num_of_samples)

# create metric with default setting
lpips_metric = pyiqa.create_metric('lpips').to(device)
psnr_metric = pyiqa.create_metric('psnr').to(device)
gmsd_metric = pyiqa.create_metric('gmsd').to(device)
ssim_metric = pyiqa.create_metric('ssim').to(device)
niqe_metric = pyiqa.create_metric('niqe').to(device)

# check if lower better or higher better
print('LPIPS lower is better: {}'.format(lpips_metric.lower_better))
print('PSNR lower is better: {}'.format(psnr_metric.lower_better))
print('GMSD lower is better: {}'.format(gmsd_metric.lower_better))
print('SSIM lower is better: {}'.format(ssim_metric.lower_better))
print('NIQE lower is better: {}'.format(niqe_metric.lower_better))

for i in range(len(clean_data)):
    clean, _ = clean_data[i]
    poison, _ = poison_data[i]

    clean = transform_test(clean).to(device)
    poison = transform_test(poison).to(device)

    clean = torch.unsqueeze(clean, dim=0)
    poison = torch.unsqueeze(poison, dim=0)
    # img_tensor_x/y: (N, 3, H, W), RGB, 0 ~ 1

    print("=================================================> \n")

    lpips_score = lpips_metric(clean, poison).detach().cpu().item()
    print("LPIPS for image {} is: {:.4f} \n".format(i, lpips_score))
    lpips_list.append(lpips_score)

    psnr_score = psnr_metric(clean, poison).detach().cpu().item()
    print("PSNR for image {} is: {:.4f} \n".format(i, psnr_score))
    psnr_list.append(psnr_score)

    gmsd_score = gmsd_metric(clean, poison).detach().cpu().item()
    print("GMSD for image {} is: {:.4f} \n".format(i, gmsd_score))
    gmsd_list.append(gmsd_score)

    ssim_score = ssim_metric(clean, poison).detach().cpu().item()
    print("ssim for image {} is: {:.4f} \n".format(i, ssim_score))
    ssim_list.append(ssim_score)

    # niqe_clean = niqe_metric(clean).detach().cpu().item()
    # print("NIQE for clean image {} is: {:.4f} \n".format(i, niqe_clean))
    # niqe_clean_list.append(niqe_clean)
    #
    # niqe_poison = niqe_metric(poison).detach().cpu().item()
    # print("NIQE for poison image {} is: {:.4f} \n".format(i, niqe_poison))
    # niqe_poison_list.append(niqe_poison)

print("=================================================> \n")

print("Mean LPIPS: {:.4f} | Max lpips: {:.4f} | Min lpips: {:.4f} | variance: {:.4f}\n".
      format(statistics.mean(lpips_list), max(lpips_list), min(lpips_list), statistics.variance(lpips_list)))

print("Mean PSNR: {:.4f} | Max PSNR: {:.4f} | Min PSNR: {:.4f} | variance: {:.4f}\n".
      format(statistics.mean(psnr_list), max(psnr_list), min(psnr_list), statistics.variance(psnr_list)))

print("Mean GMSD: {:.4f} | Max GMSD: {:.4f} | Min GMSD: {:.4f} | variance: {:.4f}\n".
      format(statistics.mean(gmsd_list), max(gmsd_list), min(gmsd_list), statistics.variance(gmsd_list)))

print("Mean SSIM: {:.4f} | Max SSIM: {:.4f} | Min SSIM: {:.4f} | variance: {:.4f}\n".
      format(statistics.mean(ssim_list), max(ssim_list), min(ssim_list), statistics.variance(ssim_list)))

# print("Clean IMAGE: \nMean NIQE: {:.4f} | Max NIQE: {:.4f} | Min NIQE: {:.4f} ".
#       format(statistics.mean(niqe_clean_list), max(niqe_clean_list), min(niqe_clean_list)))
#
# print("Poison IMAGE: \nMean NIQE: {:.4f} | Max NIQE: {:.4f} | Min NIQE: {:.4f} ".
#       format(statistics.mean(niqe_poison_list), max(niqe_poison_list), min(niqe_poison_list)))