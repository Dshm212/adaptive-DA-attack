import torchvision
import cv2
import numpy as np
import os
from function import setDir

cifar10_train = torchvision.datasets.CIFAR10(
    './', train=True, download=True
)

root = '/mnt/ssd4/chaohui/datasets/CIFAR10/'
train_dir = root + '/clean_train/'
setDir(train_dir)

txt_name = root + '/clean_train.txt'
if os.path.exists(txt_name):
    os.remove(txt_name)

f = open(txt_name, 'w')

print('train set:', len(cifar10_train))

for i, (img, label) in enumerate(cifar10_train):
    img_path = train_dir + str(i) + ".png"
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, img)
    f.write(img_path + ' ' + str(label) + '\n')
f.close()
