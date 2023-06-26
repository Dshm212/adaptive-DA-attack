import copy
import cv2
import numpy as np
import os
import torchvision
from poisoning import image_poisoning
from tqdm import tqdm
from function import setDir


def backdoor_train_image(args):
    data_root = args.data

    backdoor_train = os.path.join(data_root, 'backdoor_train')
    instance_dir = os.path.join(data_root, 'train_instance')
    poison_dir = os.path.join(data_root, 'train_poison')

    Cifar_backdoor_train = torchvision.datasets.CIFAR10('./', train=True, download=True)

    train_txt = os.path.join(data_root, 'backdoor_train.txt')
    instance_txt = os.path.join(data_root, 'train_instance.txt')
    poison_txt = os.path.join(data_root, 'train_poison.txt')

    setDir(backdoor_train)
    setDir(instance_dir)
    setDir(poison_dir)

    if os.path.exists(train_txt):
        os.remove(train_txt)
    if os.path.exists(instance_txt):
        os.remove(instance_txt)
    if os.path.exists(poison_txt):
        os.remove(poison_txt)

    f = open(train_txt, 'w')
    c = open(instance_txt, 'w')
    p = open(poison_txt, 'w')

    num_of_samples = len(Cifar_backdoor_train)
    print('train set:', num_of_samples)

    count_image = 0
    count = 0
    count_dirty = 0
    injection_ratio = args.ir

    selected_clean_list = []

    pbar = tqdm(total=len(Cifar_backdoor_train), desc='Backdoor training dataset generation')

    for i in range(num_of_samples):
        (img, label) = Cifar_backdoor_train.__getitem__(i)
        img_path = os.path.join(backdoor_train, str(i) + args.ext)
        img = np.array(img)

        if (label == args.target_label) & (count_dirty < int(injection_ratio * num_of_samples)):
            selected_clean_list.append(img)
            count_dirty += 1
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img)
            f.writelines(img_path + ' ' + str(label) + '\n')

        count += 1
        count_image += 1
        pbar.update(1)

    mini_batch_size = args.bs
    count_dirty = 0
    for j in range(0, len(selected_clean_list), mini_batch_size):
        mini_batch = selected_clean_list[j: min(j + mini_batch_size, len(selected_clean_list))]
        copy_batch = copy.deepcopy(mini_batch)
        poisoned_batch = image_poisoning(copy_batch, True, args)
        for k in range(len(mini_batch)):
            clean = mini_batch[k]
            poison = poisoned_batch[k]

            clean = cv2.cvtColor(clean, cv2.COLOR_RGB2BGR)
            poison = cv2.cvtColor(poison, cv2.COLOR_RGB2BGR)

            label = args.target_label

            img_path = os.path.join(backdoor_train, str(count_image) + args.ext)
            instance_path = os.path.join(instance_dir, str(count_dirty) + args.ext)
            poison_path = os.path.join(poison_dir, str(count_dirty) + args.ext)

            cv2.imwrite(img_path, poison)
            cv2.imwrite(instance_path, clean)
            cv2.imwrite(poison_path, poison)

            f.writelines(img_path + ' ' + str(label) + '\n')
            c.writelines(instance_path + ' ' + str(label) + '\n')
            p.writelines(poison_path + ' ' + str(label) + '\n')

            count_dirty += 1
            count_image += 1

    f.close()
    c.close()
    p.close()
    pbar.close()

    print('{} training images are poisoned!'.format(count_dirty))


if __name__ == "__main__":
    import config
    args = config.get_arguments().parse_args()
    backdoor_train_image(args=args)
