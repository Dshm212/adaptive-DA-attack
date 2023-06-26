import torchvision
import cv2
import numpy as np
import os
from function import MyDataset, setDir
from poisoning import image_poisoning
from tqdm import tqdm


def backdoor_test_image(args):
    data_root = args.data

    backdoor_test = os.path.join(data_root, 'backdoor_test')
    instance_dir = os.path.join(data_root, 'test_instance')
    poison_dir = os.path.join(data_root, 'test_poison')

    Cifar_backdoor_test = torchvision.datasets.CIFAR10('./', train=False, download=True)

    test_txt = os.path.join(data_root, 'backdoor_test.txt')
    instance_txt = os.path.join(data_root, 'test_instance.txt')
    poison_txt = os.path.join(data_root, 'test_poison.txt')

    setDir(backdoor_test)
    setDir(instance_dir)
    setDir(poison_dir)

    if os.path.exists(test_txt):
        os.remove(test_txt)
    if os.path.exists(instance_txt):
        os.remove(instance_txt)
    if os.path.exists(poison_txt):
        os.remove(poison_txt)

    f = open(test_txt, 'w')
    c = open(instance_txt, 'w')
    p = open(poison_txt, 'w')

    pbar = tqdm(total=len(Cifar_backdoor_test), desc='Backdoor test dataset generation')
    count_dirty = 0

    for i, (img, label) in enumerate(Cifar_backdoor_test):
        # if label != args.target_label:
        # if label == 0:
        if (label != args.target_label) & (i <= 2000):

            img_path = os.path.join(backdoor_test, str(count_dirty) + args.ext)
            instance_path = os.path.join(instance_dir, str(count_dirty) + args.ext)
            poison_path = os.path.join(poison_dir, str(count_dirty) + args.ext)

            img = np.array(img)
            poison = image_poisoning([img], False, args)[0]

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            poison = cv2.cvtColor(poison, cv2.COLOR_RGB2BGR)

            cv2.imwrite(img_path, poison)
            cv2.imwrite(instance_path, img)
            cv2.imwrite(poison_path, poison)

            f.writelines(img_path + ' ' + str(args.target_label) + '\n')
            c.writelines(instance_path + ' ' + str(label) + '\n')
            p.writelines(poison_path + ' ' + str(args.target_label) + '\n')

            count_dirty += 1

        pbar.update(1)

    f.close()
    c.close()
    p.close()
    pbar.close()


if __name__ == "__main__":
    import config
    args = config.get_arguments().parse_args()
    backdoor_test_image(args=args)
