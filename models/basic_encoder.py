import os.path

import torch
import torch.nn as nn
from models.dual_model_mae import mae_vit_base_patch16
from models.tinymim import tinymim_vit_tiny_patch16
from utils.pos_embed import interpolate_pos_embed, interpolate_pos_encoding
from torchvision.transforms.functional import normalize


class PrimaryDeblurVal(nn.Module):
    def __init__(self,depth=4):
        super(PrimaryDeblurVal, self).__init__()

        img_size = 512
        self.pretrain_mae = mae_vit_base_patch16(img_size=img_size)
        self.pretrain_mae.patch_embed.proj.stride = 4
        self.layer_depth = depth

        del self.pretrain_mae.decoder_blocks, self.pretrain_mae.decoder_pred, self.pretrain_mae.decoder_norm
        del self.pretrain_mae.decoder_embed, self.pretrain_mae.decoder_pos_embed
        del self.pretrain_mae.blocks[5:]
        pos_embed = interpolate_pos_encoding( 3969, 768, self.pretrain_mae, 16, (8,8), 512, 512)
        self.pretrain_mae.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

    def forward(self, recover_img, gt=None, is_train=False):
        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        predict_embed, gt_embed, ids = self.pretrain_mae(recover_img, gt, 0.0)

        return predict_embed[:self.layer_depth]


class PrimaryDeblurTest(nn.Module):
    def __init__(self,depth=4):
        super(PrimaryDeblurTest, self).__init__()
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.layer_depth = depth
        self.backup_pos = None
        self.token_num =  961 
        self.img_size = 320 
        self.mae_embed = 768

    def repos(self, model):
        pos_embed = interpolate_pos_encoding(self.token_num, self.mae_embed, model.pretrain_mae, 16, (10, 10), self.img_size, self.img_size)
        self.backup_pos = model.pretrain_mae.pos_embed.clone()
        model.pretrain_mae.pos_embed = nn.Parameter(pos_embed, requires_grad=False)

    def setbackpos(self, model):
        pos_embed = self.backup_pos
        model.pretrain_mae.pos_embed = nn.Parameter(pos_embed, requires_grad=False)

    def forward_repos(self, recover_img, model, kqv=False):
        old_pos_embed = model.pretrain_mae.pos_embed
        model.pretrain_mae.pos_embed = nn.Parameter(model.pos_embed_sparse, requires_grad=False)
        model.pretrain_mae.pos_embed.data = model.pretrain_mae.pos_embed.data.to(old_pos_embed.device)
        model.pretrain_mae.patch_embed.proj.stride = 16

        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        predict_embed, gt_embed, ids = model.pretrain_mae(recover_img, None, 0.0, kqv=kqv)
        model.pretrain_mae.pos_embed = old_pos_embed
        model.pretrain_mae.patch_embed.proj.stride = 10

        return predict_embed[:self.layer_depth]

    def forward(self, recover_img, model, kqv=False):
        self.repos(model)
        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        predict_embed, gt_embed, ids = model.pretrain_mae(recover_img, None, 0.0, kqv=kqv)

        return predict_embed[:self.layer_depth]


class PrimaryLightDeblur(nn.Module):
    def __init__(self,depth=4):
        super(PrimaryLightDeblur, self).__init__()

        img_size = 224
        self.pretrain_mae = tinymim_vit_tiny_patch16(img_size=img_size)
        self.pretrain_mae.patch_embed.proj.stride = 10
        self.layer_depth = depth

        del self.pretrain_mae.blocks[5:]

        pos_embed = interpolate_pos_encoding(961, 192, self.pretrain_mae, 16, (10, 10), 316, 316)
        self.pos_embed_sparse = interpolate_pos_encoding(400, 192, self.pretrain_mae, 16, (16, 16), 320, 320)
        self.pos_embed_sparse = interpolate_pos_encoding(900, 192, self.pretrain_mae, 16, (7, 7), 224, 224)
        self.pos_embed_coarse = interpolate_pos_encoding(2601, 192, self.pretrain_mae, 16, (5, 5), 266, 266)
        self.pos_embed_coarse = interpolate_pos_encoding(961, 192, self.pretrain_mae, 16, (10, 10), 316, 316)

        self.pretrain_mae.pos_embed = nn.Parameter(pos_embed, requires_grad=False)

        self.init_pos = False

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

    def forward_repos(self, recover_img, gt=None, is_train=False, kqv=False):
        old_pos_embed = self.pretrain_mae.pos_embed
        self.pretrain_mae.pos_embed = nn.Parameter(self.pos_embed_sparse, requires_grad=False)
        self.pretrain_mae.pos_embed.data = self.pretrain_mae.pos_embed.data.to(old_pos_embed.device)
        self.pretrain_mae.patch_embed.proj.stride = 16

        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        gt_img = normalize(gt, self.normalize_mean, self.normalize_std)
        predict_embed, gt_embed, ids = self.pretrain_mae(recover_img, gt_img, 0.0, kqv=kqv)
        self.pretrain_mae.pos_embed = old_pos_embed
        self.pretrain_mae.patch_embed.proj.stride = 10

        return predict_embed[:self.layer_depth], gt_embed[:self.layer_depth]


    def forward(self, recover_img, gt=None, is_train=False, kqv=False):
        if not self.init_pos:
            old_pos_embed = self.pretrain_mae.pos_embed
            self.pretrain_mae.pos_embed = nn.Parameter(self.pos_embed_coarse, requires_grad=False)
            self.pretrain_mae.pos_embed.data = self.pretrain_mae.pos_embed.data.to(old_pos_embed.device)
            self.init_pos = False
        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        gt_img = normalize(gt, self.normalize_mean, self.normalize_std)
        predict_embed, gt_embed, ids = self.pretrain_mae(recover_img, gt_img, 0.0, kqv=kqv)

        return predict_embed[:self.layer_depth], gt_embed[:self.layer_depth]


class PrimaryDeblur(nn.Module):
    def __init__(self,depth=4):
        super(PrimaryDeblur, self).__init__()

        img_size = 224
        self.pretrain_mae = mae_vit_base_patch16(img_size=img_size)
        self.pretrain_mae.patch_embed.proj.stride = 10
        self.layer_depth = depth

        del self.pretrain_mae.decoder_blocks, self.pretrain_mae.decoder_pred, self.pretrain_mae.decoder_norm
        del self.pretrain_mae.decoder_embed, self.pretrain_mae.decoder_pos_embed
        del self.pretrain_mae.blocks[5:]

        pos_embed = interpolate_pos_encoding(961, 768, self.pretrain_mae, 16, (10, 10), 320, 320)
        pos_embed = interpolate_pos_encoding(1225, 768, self.pretrain_mae, 16, (10, 10), 360, 360)
        self.pos_embed_sparse = interpolate_pos_encoding(400, 768, self.pretrain_mae, 16, (16, 16), 320, 320)
        self.pos_embed_sparse = interpolate_pos_encoding(900, 768, self.pretrain_mae, 16, (7, 7), 224, 224)
        self.pos_embed_coarse = interpolate_pos_encoding(961, 768, self.pretrain_mae, 16, (10, 10), 320, 320)

        self.pretrain_mae.pos_embed = nn.Parameter(pos_embed, requires_grad=False)

        self.init_pos = True

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

    def forward_repos(self, recover_img, gt=None, is_train=False, kqv=False):
        old_pos_embed = self.pretrain_mae.pos_embed
        self.pretrain_mae.pos_embed = nn.Parameter(self.pos_embed_sparse, requires_grad=False)
        self.pretrain_mae.pos_embed.data = self.pretrain_mae.pos_embed.data.to(old_pos_embed.device)
        self.pretrain_mae.patch_embed.proj.stride = 16

        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        gt_img = normalize(gt, self.normalize_mean, self.normalize_std)
        predict_embed, gt_embed, ids = self.pretrain_mae(recover_img, gt_img, 0.0, kqv=kqv)
        self.pretrain_mae.pos_embed = old_pos_embed
        self.pretrain_mae.patch_embed.proj.stride = 10

        return predict_embed[:self.layer_depth], gt_embed[:self.layer_depth]


    def forward(self, recover_img, gt=None, is_train=False, kqv=False):
        if not self.init_pos:
            old_pos_embed = self.pretrain_mae.pos_embed
            self.pretrain_mae.pos_embed = nn.Parameter(self.pos_embed_coarse, requires_grad=False)
            self.pretrain_mae.pos_embed.data = self.pretrain_mae.pos_embed.data.to(old_pos_embed.device)
            self.init_pos = True
        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        gt_img = normalize(gt, self.normalize_mean, self.normalize_std)
        predict_embed, gt_embed, ids = self.pretrain_mae(recover_img, gt_img, 0.0, kqv=kqv)


        return predict_embed[:self.layer_depth], gt_embed[:self.layer_depth]

