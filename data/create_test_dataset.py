from torch.utils.data import Dataset
import os
import torch
import torchvision.transforms.functional as TF
import cv2


class TestDataset(Dataset):
    def __init__(self, testopt):
        super(TestDataset, self).__init__()
        path = testopt['dataroot']

        combine_imgs = os.path.join(os.path.expanduser(path), testopt["combine_name"])
        blur_imgs = os.path.join(os.path.expanduser(path), testopt["blur_name"])

        combine_imgs = [os.path.join(combine_imgs, os_dir) for os_dir in os.listdir(combine_imgs)]
        blur_imgs = [os.path.join(blur_imgs, os_dir) for os_dir in os.listdir(blur_imgs)]

        combine_imgs.sort()
        blur_imgs.sort()

        self.uegt_imgs = {}
        for  c_img, b_img in zip(combine_imgs, blur_imgs):
            self.uegt_imgs[c_img] = [b_img]

        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.uegt_imgs)

    def pad_to_even(self, img):
        height, width, _ = img.shape

        if height < 320 or width < 320:
            new_height = max(height, 320)
            new_width = max(width, 320)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        pad_bottom = (16 - img.shape[0] % 16) % 16
        pad_right = (16 - img.shape[1] % 16) % 16
        return cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

    def __getitem__(self, index):
        c_img, b_img = self.uegt_imgs[index]
        c_img_name = c_img

        gt_img = cv2.imread(c_img, cv2.IMREAD_COLOR)
        b_img = cv2.imread(b_img[0], cv2.IMREAD_COLOR)


        gt_img = self.pad_to_even(gt_img)
        b_img = self.pad_to_even(b_img)

        gt_img = torch.tensor(gt_img / 255.0).float().permute(2, 0, 1)
        b_img = torch.tensor(b_img / 255.).float().permute(2, 0, 1)

        return gt_img, b_img, os.path.basename(c_img_name).split('.')
