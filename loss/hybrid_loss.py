import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
import torch.distributed as dist
import torch.nn.functional as F


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class SelfLearningLoss(nn.Module):
    def __init__(self,batch_size):
        super(SelfLearningLoss, self).__init__()
        self.mse_loss = MSELoss()
        self.cos_loss = nn.CosineSimilarity(dim=2)
        self.l1_loss = L1Loss()
        self.sim_coeff = 0.10
        self.std_coeff = 0.10
        self.cov_coeff = 0.01
        self.reconstruct_coeff = 1
        self.num_features = 256
        self.batch_size = batch_size * 2

    def forward(self, x, y, x_z, y_z):
        repr_loss = self.mse_loss(x_z, y_z)
        recons_loss = (1 - self.cos_loss(x,y)).mean()

        x_z = torch.cat(FullGatherLayer.apply(x_z), dim=0)
        y_z = torch.cat(FullGatherLayer.apply(y_z), dim=0)
        x_z = x_z - x_z.mean(dim=0)
        y_z = y_z - y_z.mean(dim=0)

        std_x = torch.sqrt(x_z.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y_z.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x_z.T @ x_z) / (self.batch_size - 1)
        cov_y = (y_z.T @ y_z) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
                self.sim_coeff * repr_loss
                + self.std_coeff * std_loss
                + self.cov_coeff * cov_loss
        )
        return recons_loss, loss, (self.sim_coeff * repr_loss, self.std_coeff * std_loss, self.cov_coeff * cov_loss)


class ReconstructLoss(nn.Module):
    def __init__(self, opt):
        super(ReconstructLoss, self).__init__()
        self.l1_loss = CharbonnierLoss()
        self.mse_loss = MSELoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.self_learning = SelfLearningLoss(opt['batch_size'])

    def forward(self, clsblur_embed, clsdeblur_embed, deblur_embed, guidance_embed, deblur_z, guidance_z, recover_img, gt, recover_list, gt_list, recover_mae, guidance_mae):
        losses = {}
        loss_l1 = self.l1_loss(recover_img, gt)

        recons_loss, loss_self, (repr_loss, std_loss, cov_loss) = self.self_learning(deblur_embed, guidance_embed, deblur_z, guidance_z)

        loss_self = loss_self * 1e-1

        blur_target = torch.zeros(clsblur_embed.shape[0]).to(clsblur_embed.device)
        sharp_target = torch.ones(clsdeblur_embed.shape[0]).to(clsdeblur_embed.device)

        loss_cls = self.bce_loss(clsblur_embed,blur_target) + self.bce_loss(clsdeblur_embed, sharp_target)
        loss_cls = loss_cls * 3e-1
        loss_vicreg = recons_loss + loss_cls

        loss_vicreg = loss_vicreg * 1 

        losses["self_learning/total"] = loss_vicreg
        losses["self_learning/recons"] = recons_loss
        losses["self_learning/vicreg"] = loss_self
        losses["self_learning/cls"] = loss_cls
        losses["vicreg/repr"] = repr_loss
        losses["vicreg/std"] = std_loss
        losses["vicreg/cov"] = cov_loss

        loss_vicreg = 0.2 * loss_vicreg 

        #
        losses["total_loss/final_loss"] = loss_l1 + loss_vicreg
        losses["total_loss/l1"] = loss_l1
        losses["total_loss/vicreg"] = loss_vicreg

        return losses
