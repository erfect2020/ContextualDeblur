import os.path

import torch
import torch.nn as nn
from models.dual_model_mae import mae_vit_base_patch16
from models.tinymim import tinymim_vit_tiny_patch16, tinymim_vit_tiny_patch8, tinymim_vit_small_patch16
from utils.pos_embed import interpolate_pos_embed, interpolate_pos_encoding
from torchvision.transforms.functional import normalize


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class PrimaryDeblurVal(nn.Module):
    def __init__(self,depth=4):
        super(PrimaryDeblurVal, self).__init__()

        img_size = 512
        pretrained_ckpt = './models/mae_pretrain_vit_base.pth'
        self.pretrain_mae = mae_vit_base_patch16(img_size=img_size)
        self.pretrain_mae.patch_embed.proj.stride = 4
        self.layer_depth = depth
        pretrained_ckpt = os.path.expanduser(pretrained_ckpt)
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrained_ckpt)
        checkpoint_model = checkpoint['model']
        state_dict = self.pretrain_mae.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate position embedding
        interpolate_pos_embed(self.pretrain_mae, checkpoint_model)
        self.pretrain_mae.load_state_dict(checkpoint_model, strict=False)
        for _, p in self.pretrain_mae.named_parameters():
            p.requires_grad = False

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
        self.token_num =  961 #961 #2601 #1225 #2209 #2304 #961 #1225 #2025 #1225
        self.img_size = 320 #320 #316 #520 # 360 #476 #360 #460
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
        pretrained_ckpt = './models/TinyMIM-PT-Tstar.pth'
        self.pretrain_mae = tinymim_vit_tiny_patch16(img_size=img_size)
        self.pretrain_mae.patch_embed.proj.stride = 10
        self.layer_depth = depth
        pretrained_ckpt = os.path.expanduser(pretrained_ckpt)
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrained_ckpt)
        checkpoint_model = checkpoint['model']
        state_dict = self.pretrain_mae.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        self.pretrain_mae.load_state_dict(checkpoint_model, strict=False)
        for _, p in self.pretrain_mae.named_parameters():
            p.requires_grad = True

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
        pretrained_ckpt = './models/mae_pretrain_vit_base.pth'
        self.pretrain_mae = mae_vit_base_patch16(img_size=img_size)
        self.pretrain_mae.patch_embed.proj.stride = 10
        self.layer_depth = depth
        pretrained_ckpt = os.path.expanduser(pretrained_ckpt)
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrained_ckpt)
        checkpoint_model = checkpoint['model']
        state_dict = self.pretrain_mae.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        self.pretrain_mae.load_state_dict(checkpoint_model, strict=False)
        for _, p in self.pretrain_mae.named_parameters():
            p.requires_grad = False

        del self.pretrain_mae.decoder_blocks, self.pretrain_mae.decoder_pred, self.pretrain_mae.decoder_norm
        del self.pretrain_mae.decoder_embed, self.pretrain_mae.decoder_pos_embed
        del self.pretrain_mae.blocks[5:]

        pos_embed = interpolate_pos_encoding(961, 768, self.pretrain_mae, 16, (10, 10), 320, 320)
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

