import os.path

import numpy as np
import torch
import torch.nn as nn
from models.basic_encoder import PrimaryDeblur, PrimaryDeblurTest
from timm.models.vision_transformer import Block


class ContextualEncoder(nn.Module):
    def __init__(self):
        super(ContextualEncoder, self).__init__()
        encoder_embed_dim = 768
        encoder_depth = 3
        encoder_num_heads = 16
        mlp_ratio = 4.
        norm_layer = nn.LayerNorm

        self.embed_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)])

        self.embed_norm = norm_layer(encoder_embed_dim)


        projector_layer = []
        f = [768, 256, 128]
        for i in range(len(f) - 2):
            projector_layer.append(nn.Linear(f[i], f[i + 1]))
            projector_layer.append(nn.BatchNorm1d(f[i + 1]))
            projector_layer.append(nn.ReLU(True))
        projector_layer.append(nn.Linear(f[-2], f[-1], bias=False))

        self.projector = nn.Sequential(*projector_layer)
        self.blur_parameter = nn.parameter.Parameter(torch.zeros(encoder_embed_dim))

        self.fc_norm = norm_layer(encoder_embed_dim)
        self.head = nn.Linear(encoder_embed_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y, is_train=False ):

        if is_train:
            for blk in self.embed_blocks:
                x = blk(x)
                y = blk(y)
            x = self.embed_norm(x)
            y = self.embed_norm(y)

            x_avg = x[:, 1:, :].mean(dim=1)
            x_avg = self.fc_norm(x_avg)
            x_outcome = self.head(x_avg).squeeze()

            y_avg = y[:, 1:, :].mean(dim=1)
            y_avg = self.fc_norm(y_avg)
            y_outcome = self.head(y_avg).squeeze()

            hat_y = x * self.blur_parameter
            z_x = self.projector(hat_y[:, 1:, :].mean(dim=1))
            z_y = self.projector(y[:, 1:, :].mean(dim=1))
            return x_outcome, y_outcome, hat_y, y, z_x, z_y
        else:
            for blk in self.embed_blocks:
                x = blk(x)
            x = self.embed_norm(x)
            x = x * self.blur_parameter
            return x

class RecoverDecoder(nn.Module):
    def __init__(self,decoder_depth):
        super(RecoverDecoder, self).__init__()

        decoder_embed_dim = 1024
        decoder_num_heads = 16
        mlp_ratio = 4.
        norm_layer = nn.LayerNorm
        self.decoder_depth = decoder_depth

        self.decoder_linears = nn.ModuleList([nn.Sequential(*[norm_layer(768 * 3),
                                                              nn.Linear(768 * 3, decoder_embed_dim, bias=True)])])
        self.decoder_dnorm1 = norm_layer(768)
        self.decoder_dnorm2 = norm_layer(768)
        self.decoder_anorm = norm_layer(768)

        for i in range(decoder_depth-1):
            self.decoder_linears.append(nn.Sequential(*[norm_layer(768 * 2 + decoder_embed_dim),
                                                        nn.Linear(768 * 2 + decoder_embed_dim, decoder_embed_dim, bias=True)]))

        self.embed_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])


        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 24 * 100, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, encoder, x):
        x = x[:,1:, :]

        for i in range(len(encoder)):
            encoder[i] = encoder[i][:,1:,:]

        encoder[-1] = self.decoder_dnorm1(encoder[-1])
        x = self.decoder_anorm(x)
        for i in range(self.decoder_depth):
            x = torch.cat((encoder[self.decoder_depth-1-i], encoder[self.decoder_depth-1-i], x), dim=2)
            x = self.decoder_linears[i](x)
            x = self.embed_blocks[i](x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x


class UnifyDeblur(nn.Module):
    def __init__(self):
        super(UnifyDeblur, self).__init__()
        self.layer_depth = 4
        self.primary_encoder = PrimaryDeblur(self.layer_depth)
        self.primary_encoder_val = PrimaryDeblurTest(self.layer_depth)
        self.contextual_encoder = ContextualEncoder()

        self.decoder = RecoverDecoder(self.layer_depth)

        self.outputprojr_step = nn.Conv2d(24, 3, kernel_size=13, stride=1, padding=11, padding_mode='reflect')
        self.inputecoding_step = nn.Conv2d(6, 24, kernel_size=11, stride=1, padding=0)


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 16
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def unpatchifyc(self, x, c=3, p=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = p
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, blur, guidance=None, is_train= False, no_ttsc=True):
        if is_train:
            blur_embed, guidance_embed = self.primary_encoder(blur, guidance, True)

            clsblur_embed, clsdeblur_embed, deblur_embed, guidance_embed_rep, deblur_z, guidance_z = self.contextual_encoder(blur_embed[-1], guidance_embed[-1], True)

            recover = self.decoder(blur_embed, deblur_embed)
            recover = self.unpatchifyc(recover[:,:,:], c=24, p=10)

            blur_inp = self.inputecoding_step(blur)
            recover = recover + blur_inp

            recover = self.outputprojr_step(recover)

            recover_mae, guidance_mae = None, None

            return  self.contextual_encoder.blur_parameter, clsblur_embed, clsdeblur_embed, deblur_embed, guidance_embed_rep, deblur_z, guidance_z, recover, recover_mae, guidance_mae
        else:
            # hyperparameter
            stride = 16
            blur_grids = TestGrids(blur)
            blur_grids.lq = blur_grids.lq.cpu()
            blur_grids.update_device()
            blur_grids.grids()
            test_bs = 16
            total_num = blur_grids.lq.size(0)
            output = np.zeros((total_num, 24, 310, 310))
            import gc
            from tqdm import tqdm
            for i in range(0, total_num, test_bs):
                end_i = min(total_num, i+test_bs)
                blur_embed = self.primary_encoder_val(blur_grids.lq[i:end_i].to('cuda:0'), self.primary_encoder)
                deblur_embed = self.contextual_encoder(blur_embed[-1], None, False)
                recover = self.decoder(blur_embed, deblur_embed)
                recover = self.unpatchifyc(recover[:, :, :], c=24, p=10)
                output[i:end_i] = recover.cpu().detach().numpy()

            blur_grids.output = output
            blur_grids.grids_inverse()
            recover = blur_grids.output
            recover = recover[:,:,5:-5, 5:-5].to('cuda:0')
            blur_inp = self.inputecoding_step(blur)
            recover = recover + blur_inp
            recover = self.outputprojr_step(recover)
            return recover


class TestGrids():
    def __init__(self,input_img):
        super(TestGrids, self).__init__()
        self.lq = input_img
        self.output =None
        self.device = self.lq.device
    def update_device(self):
        self.device = self.lq.device

    def grids(self):
        b, c, h, w = self.lq.size()
        self.original_size = [self.lq.size(0), 24, self.lq.size(2), self.lq.size(3)]
        assert b == 1
        crop_size = 320
        self.img_crop_size = crop_size
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)

        step_i = step_j = 47 
        parts = []
        idxes = []
        fea_idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                parts.append(self.lq[:, :, i:i + crop_size, j:j + crop_size])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)

        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = 310
        self.fea_crop_size = crop_size


        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i'] + 5
            j = each_idx['j'] + 5

            preds[0, :, i:i + crop_size, j:j + crop_size] += self.output[cnt, :, :, :]
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        count_mt = torch.where(count_mt < 1e-6, torch.tensor(1e-6), count_mt)
        self.output = preds / count_mt
        self.lq = self.origin_lq
