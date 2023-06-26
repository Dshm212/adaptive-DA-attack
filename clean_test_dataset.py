import torchvision
import cv2
import numpy as np
import os
from function import setDir

cifar10_clean_test = torchvision.datasets.CIFAR10(
    './', train=False, download=True
)

root = '/mnt/ssd4/chaohui/datasets/CIFAR10/'

test_dir = root + '/clean_test/'
setDir(test_dir)

txt_name = root + '/clean_test.txt'
if os.path.exists(txt_name):
    os.remove(txt_name)
    print('txt file removed')

f = open(txt_name, 'w')

print('test set:', len(cifar10_clean_test))

for i, (img, label) in enumerate(cifar10_clean_test):
    img_path = test_dir + str(i) + ".png"
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # save images
    cv2.imwrite(img_path, img)
    f.write(img_path + ' ' + str(label) + '\n')
f.close()
