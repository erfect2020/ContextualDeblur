import time
from torch.utils.data import Dataset
import os
import json
import torch
from torchvision.transforms import Compose, RandomCrop,RandomHorizontalFlip, RandomVerticalFlip, Normalize, Resize
import cv2



class TrainDataset(Dataset):
    def __init__(self, trainopt):
        super(TrainDataset, self).__init__()
        paths = [("defocus-single", trainopt['defocusdataroot'])]
        self.image_size = trainopt['image_size']
        self.batch_size = trainopt['iter_size']
        self.max_iter = trainopt["max_iter"]
        self.epoch_num = 0
        self.data_augment = Compose(
            [
                RandomCrop(self.image_size),
                RandomHorizontalFlip(),
                # RandomVerticalFlip()
                # RandomRotation()
            ]
        )
        self.preprocess = Compose(
            [   Resize(224),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
        )

        trainpairs_forreading = str(trainopt['trainpairs'])
        if not os.path.exists(trainpairs_forreading):
            self.uegt_imgs = {}
            for path_name, path in paths:
                left_imgs = os.path.join(os.path.expanduser(path), trainopt["defocus_blur_name"])
                right_imgs = os.path.join(os.path.expanduser(path), trainopt["defocus_blur_name"])
                blur_imgs = os.path.join(os.path.expanduser(path), trainopt["defocus_blur_name"])
                combine_imgs = os.path.join(os.path.expanduser(path), trainopt["defocus_combine_name"])

                left_imgs = [os.path.join(left_imgs, os_dir) for os_dir in os.listdir(left_imgs)]
                right_imgs = [os.path.join(right_imgs, os_dir) for os_dir in os.listdir(right_imgs)]
                combine_imgs = [os.path.join(combine_imgs, os_dir) for os_dir in os.listdir(combine_imgs)]
                blur_imgs = [os.path.join(blur_imgs, os_dir) for os_dir in os.listdir(blur_imgs)]

                left_imgs.sort()
                right_imgs.sort()
                combine_imgs.sort()
                blur_imgs.sort()

                for l_img, r_img, c_img, b_img in zip(left_imgs, right_imgs, combine_imgs, blur_imgs):
                    self.uegt_imgs[c_img+'@'+path_name] = [path_name, l_img, r_img, b_img, c_img]
            with open(trainpairs_forreading, 'w') as f:
                json.dump(self.uegt_imgs, f)
        else:
            with open(trainpairs_forreading, 'r') as f:
                self.uegt_imgs = json.load(f)
        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])
        self.set_dafault_img()

    def __len__(self):
        return len(self.uegt_imgs)

    def random_augmentation(self, under):
        c, w, h = under.shape
        w_start = w - self.image_size
        h_start = h - self.image_size

        random_w = 1 if w_start <= 1 else torch.randint(low=1, high=w_start, size=(1, 1)).item()
        random_h = 1 if h_start <= 1 else torch.randint(low=1, high=h_start, size=(1, 1)).item()
        return random_w, random_h

    def set_dafault_img(self):
        l_img = "xxx/CanonDeblur/dd_dp_dataset_canon_patch/train_l/source/00000.png"
        r_img = "xxx/CanonDeblur/dd_dp_dataset_canon_patch/train_r/source/00000.png"
        b_img = "xxx/CanonDeblur/dd_dp_dataset_canon_patch/train_c/source/00000.png"
        c_img = "xxx/CanonDeblur/dd_dp_dataset_canon_patch/train_c/target/00000.png"
        self.default_limg = torch.tensor(cv2.imread(l_img,-1)/ 65535.).float().permute(2, 0, 1)
        self.default_rimg = torch.tensor(cv2.imread(r_img,-1)/ 65535.).float().permute(2, 0, 1)
        self.default_bimg = torch.tensor(cv2.imread(b_img,-1)/ 65535.).float().permute(2, 0, 1)
        self.default_gtimg = torch.tensor(cv2.imread(c_img,-1)/ 65535.).float().permute(2, 0, 1)

    def __getitem__(self, index):
        c_img_type, (blur_type, l_img, r_img, b_img, c_img) = self.uegt_imgs[index]
        # print(c_img, l_img, r_img)

        try:
            l_img = cv2.imread(l_img, cv2.IMREAD_COLOR) #cv2.IMREAD_COLOR
            r_img = cv2.imread(r_img, cv2.IMREAD_COLOR)
            gt_img = cv2.imread(c_img, cv2.IMREAD_COLOR)
            b_img = cv2.imread(b_img, cv2.IMREAD_COLOR)

            b_prompts = ""
        except:
            print('cureent exception', c_img, c_img_type)
            # time.sleep(0.5)
            # return None

        try:
            l_img = torch.tensor(l_img / 255.).float().permute(2, 0, 1)
            r_img = torch.tensor(r_img / 255.).float().permute(2, 0, 1)
            gt_img = torch.tensor(gt_img / 255.).float().permute(2, 0, 1)
            b_img = torch.tensor(b_img / 255.).float().permute(2, 0, 1)
        except Exception as e:
            print("Current exception", c_img, c_img_type)
            time.sleep(0.5)
            print("error",e)
            l_img = self.default_limg
            r_img = self.default_rimg
            b_img = self.default_bimg
            gt_img = self.default_gtimg
            # return None

        try:
            combine_im = self.data_augment(torch.cat((l_img, r_img, b_img, gt_img), dim=0))
            l_img, r_img, b_img, gt_img = combine_im[:3,:,:], combine_im[3:6,:,:], combine_im[6:9,:,:], combine_im[9:,:,:]
            prompt_image = self.preprocess(b_img)
        except:
            print("process wrong!")
            # return None

        return l_img, r_img, gt_img, b_img, { 'img':prompt_image}


class ValDataset(Dataset):
    def __init__(self, valopt):
        super(ValDataset, self).__init__()
        path = valopt['dataroot']
        left_imgs = os.path.join(os.path.expanduser(path), valopt["left_name"])
        right_imgs = os.path.join(os.path.expanduser(path), valopt["right_name"])
        combine_imgs = os.path.join(os.path.expanduser(path), valopt["combine_name"])
        blur_imgs = os.path.join(os.path.expanduser(path), valopt["blur_name"])

        left_imgs = [os.path.join(left_imgs, os_dir) for os_dir in os.listdir(left_imgs)]
        right_imgs = [os.path.join(right_imgs, os_dir) for os_dir in os.listdir(right_imgs)]
        combine_imgs = [os.path.join(combine_imgs, os_dir) for os_dir in os.listdir(combine_imgs)]
        blur_imgs = [os.path.join(blur_imgs, os_dir) for os_dir in os.listdir(blur_imgs)]
        left_imgs.sort()
        right_imgs.sort()
        combine_imgs.sort()
        blur_imgs.sort()

        self.uegt_imgs = {}
        for l_img, r_img, c_img, b_img in zip(left_imgs, right_imgs, combine_imgs, blur_imgs):
            self.uegt_imgs[c_img] = [l_img, r_img, b_img]

        self.uegt_imgs = [(key, values) for key, values in self.uegt_imgs.items()]
        self.uegt_imgs = sorted(self.uegt_imgs, key=lambda x: x[0])
        self.set_dafault_img()

    def __len__(self):
        return len(self.uegt_imgs)

    def set_dafault_img(self):
        c_img, (l_img, r_img, b_img) = self.uegt_imgs[0]
        self.default_limg = torch.tensor(cv2.imread(l_img,-1)/ 65535.).float().permute(2, 0, 1)
        self.default_rimg = torch.tensor(cv2.imread(r_img,-1)/ 65535.).float().permute(2, 0, 1)
        self.default_bimg = torch.tensor(cv2.imread(b_img,-1)/ 65535.).float().permute(2, 0, 1)
        self.default_gtimg = torch.tensor(cv2.imread(c_img,-1)/ 65535.).float().permute(2, 0, 1)
        self.default_prompts = 'The dual-pixel image with defocus blur'

    def __getitem__(self, index):
        c_img, (l_img, r_img, b_img) = self.uegt_imgs[index]
        # print(c_img, l_img, r_img)
        c_img_name = c_img

        try:

            l_img = cv2.imread(l_img, cv2.IMREAD_COLOR)
            r_img = cv2.imread(r_img, cv2.IMREAD_COLOR)
            gt_img = cv2.imread(c_img, cv2.IMREAD_COLOR)
            b_img = cv2.imread(b_img, cv2.IMREAD_COLOR)

            l_img = torch.tensor(l_img / 255.).float().permute(2, 0, 1)
            r_img = torch.tensor(r_img / 255.).float().permute(2, 0, 1)
            gt_img = torch.tensor(gt_img / 255.).float().permute(2, 0, 1)
            b_img = torch.tensor(b_img / 255.).float().permute(2, 0, 1)


            b_prompts = 'The dual-pixel image with defocus blur'
        except:
            print("trhow a except")
            l_img, r_img, gt_img, b_img = self.default_limg, self.default_rimg, self.default_gtimg, self.default_bimg
            b_prompts = self.default_prompts
            # return None
        return l_img, r_img, gt_img, b_img, {'text':b_prompts}, os.path.basename(c_img_name).split('.')

