import copy

import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
from skimage.filters import gaussian
import albumentations as A
from skimage import color
import random
import os
import math


# conduct gaussian blurring on channel 0
def Blur(img, sigma):
    img = np.array(img, dtype="float64")
    img[:, :, 0] = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)[:, :, 0]
    img = np.clip(img, 0, 255)
    img = np.array(img, dtype="uint8")
    return img


# conduct brightness & contrast adjustment on channel 1
def BrightnessContrast(img, limit):
    tmp = copy.deepcopy(img)
    tmp = np.array(tmp, dtype="uint8")

    brightness_limit = -limit * 1.2
    contrast_limit = limit
    transform = A.RandomBrightnessContrast(
        brightness_limit=(brightness_limit, brightness_limit),
        contrast_limit=(contrast_limit, contrast_limit),
        brightness_by_max=False, p=1
    )
    tmp[:, :, 1] = transform(image=tmp)["image"][:, :, 1]

    return tmp


# conduct kernel smoothing on channel 2
def Smooth(img, size):
    img = np.array(img, dtype="float64")
    kernel = np.ones((size, size), np.float64) / (size * size)
    img[:, :, 2] = cv2.filter2D(img, -1, kernel)[:, :, 2]
    img = np.clip(img, 0, 255)
    img = np.array(img, dtype="uint8")

    return img


def patch(img):
    tmp = copy.deepcopy(img)
    trigger_path = "/mnt/ssd4/chaohui/backdoor_projects/CLBA/trigger.png"
    trigger = cv2.imread(trigger_path)
    trigger = cv2.cvtColor(trigger, cv2.COLOR_BGR2RGB)
    height = img.shape[0]
    width = img.shape[1]
    trigger_height = round(height / 8)
    trigger_width = round(width / 8)
    trigger = cv2.resize(trigger, (trigger_height, trigger_width))
    tmp[height - trigger_height:height, width - trigger_width:width, :] = trigger
    return tmp


def random_patch(img):
    tmp = copy.deepcopy(img)
    trigger_path = "/mnt/ssd4/chaohui/backdoor_projects/CLBA/trigger.png"
    trigger = cv2.imread(trigger_path)
    trigger = cv2.cvtColor(trigger, cv2.COLOR_BGR2RGB)
    height = img.shape[0]
    width = img.shape[1]
    trigger_height = round(height / 4)
    trigger_width = round(width / 4)
    trigger = cv2.resize(trigger, (trigger_height, trigger_width))
    start_height = random.randint(0, height - trigger_height - 1)
    start_width = random.randint(0, width - trigger_width - 1)
    tmp[start_height:start_height + trigger_height, start_width:start_width + trigger_width, :] = trigger
    return tmp


def sinusoidal(img, delta):
    img_shape = img.shape
    m = img_shape[1]
    f = 6
    signal = np.zeros((img_shape[0], img_shape[1]), dtype="float64")

    for j in range(m):
        signal[:, j] = math.sin(2 * math.pi * (j + 1) * f / m)
    signal *= delta

    img = np.array(img, dtype="float64")
    img[:, :, 0] = signal + img[:, :, 0]
    img[:, :, 1] = signal + img[:, :, 1]
    img[:, :, 2] = signal + img[:, :, 2]

    img = np.clip(img, 0, 255)
    img = np.array(img, dtype="uint8")

    return img


def reflection(img, alpha):
    def blending_operation(ori, ref_path, alpha):
        ref = cv2.imread(ref_path)
        h, w = ori.shape[:2]
        ref = cv2.resize(ref, (w, h))
        ref = cv2.GaussianBlur(ref, ksize=(0, 0), sigmaX=1, sigmaY=1)
        img_b = np.uint8(np.clip((1 - alpha) * ori / 255. + alpha * ref / 255., 0, 1) * 255)

        return img_b

    reflection_root = "/mnt/ssd4/chaohui/datasets/Pascal_VOC_2007/ref_images/"
    num = 50

    ref_path = os.path.join(reflection_root, str(num) + ".png")
    reflected = blending_operation(img, ref_path, alpha)

    return reflected


def CH0(img):
    tmp = copy.deepcopy(img)
    tmp = np.array(tmp, dtype="uint8")

    transform = A.Sharpen(
        alpha=(0.2, 0.2),
        lightness=(0.5, 0.5),
        p=1
    )
    tmp[:, :, 0] = transform(image=tmp)["image"][:, :, 0]

    return tmp


def CH1(img):
    tmp = copy.deepcopy(img)
    tmp = np.array(tmp, dtype="uint8")

    transform = A.RandomGamma(
        gamma_limit=(90, 90),
        p=1
    )

    tmp[:, :, 1] = transform(image=tmp)["image"][:, :, 1]

    return tmp


def CH2(img):
    tmp = copy.deepcopy(img)
    tmp = np.array(tmp, dtype="uint8")

    transform = A.MedianBlur(
        blur_limit=5,
        p=1
    )
    tmp[:, :, 2] = transform(image=tmp)["image"][:, :, 2]

    return tmp
