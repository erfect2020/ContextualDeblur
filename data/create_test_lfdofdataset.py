from torch.utils.data import Dataset
import os
import torch
import torchvision.transforms.functional as TF
import cv2


class TestDataset(Dataset):
    def __init__(self, testopt):
        super(TestDataset, self).__init__()
        path = testopt['dataroot']

        left_imgs = os.path.join(os.path.expanduser(path), testopt["left_name"])
        right_imgs = os.path.join(os.path.expanduser(path), testopt["right_name"])
        combine_imgs = os.path.join(os.path.expanduser(path), testopt["combine_name"])
        blur_imgs = os.path.join(os.path.expanduser(path), testopt["blur_name"])

        input_image_names = []
        gt_image_names = []
        for sub_dir in os.listdir(blur_imgs):
            sub_dir_path = os.path.join(blur_imgs, sub_dir)
            if os.path.isdir(sub_dir_path):
                # ???????????????????
                for image_file in os.listdir(sub_dir_path):
                    if image_file.endswith(".png") or image_file.endswith(".jpg"):  # ?????????
                        input_image_names.append(os.path.join(sub_dir_path, image_file))
                        # ??ground truth????????????
                        gt_image_name = sub_dir + ".png"  # ?????????
                        gt_image_names.append(os.path.join(combine_imgs, gt_image_name))
        left_imgs = input_image_names.copy()
        right_imgs = input_image_names.copy()
        combine_imgs = gt_image_names.copy()
        blur_imgs = input_image_names.copy()
        left_imgs.sort()
        right_imgs.sort()
        combine_imgs.sort()
        blur_imgs.sort()

        self.uegt_imgs = {}
        for l_img, r_img, c_img, b_img in zip(left_imgs, right_imgs, combine_imgs, blur_imgs):
            self.uegt_imgs[b_img] = [l_img, r_img, b_img, c_img]

        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])

    def __len__(self):
        return len(self.uegt_imgs)

    def __getitem__(self, index):
        _, (l_img, r_img, b_img, c_img) = self.uegt_imgs[index]
        c_img_name = b_img

        l_img = cv2.imread(l_img, cv2.IMREAD_COLOR)
        r_img = cv2.imread(r_img, cv2.IMREAD_COLOR)
        gt_img = cv2.imread(c_img, cv2.IMREAD_COLOR)
        b_img = cv2.imread(b_img, cv2.IMREAD_COLOR)


        l_img = torch.tensor(l_img / 255.0).float().permute(2, 0, 1)
        r_img = torch.tensor(r_img / 255.0).float().permute(2, 0, 1)
        gt_img = torch.tensor(gt_img / 255.0).float().permute(2, 0, 1)
        b_img = torch.tensor(b_img / 255.).float().permute(2, 0, 1)

        return l_img, r_img, gt_img, b_img, os.path.basename(c_img_name).split('.')
