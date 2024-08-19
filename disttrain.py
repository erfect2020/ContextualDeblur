import argparse
import os
import torch
import logging
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler, AdamW, SGD, RMSprop, LBFGS, Adadelta

from tqdm import tqdm
from utils import util, build_code_arch

from data.create_blurmix_trainval_dataset import TrainDataset, ValDataset, mixup_data
from models.ContextualDeblur import  UnifyDeblur


import numpy as np

from loss.hybrid_loss import ReconstructLoss

from torchvision.transforms import ToPILImage
from PIL import Image
import math


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Unify Deblur: Path to option ymal file.')
parser.add_argument('--local_rank', default=-1, type=int)
train_args = parser.parse_args()

opt, resume_state = build_code_arch.build_resume_state(train_args)
opt, logger, tb_logger = build_code_arch.build_logger(opt, train_args)

def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


for phase, dataset_opt in opt['dataset'].items():
    if phase == 'train':
        train_dataset = TrainDataset(dataset_opt)
        train_loader = DataLoader(
            train_dataset, batch_size=dataset_opt['batch_size'], shuffle=True,
            prefetch_factor=4 ,num_workers=dataset_opt['workers'], pin_memory=True, collate_fn=my_collate_fn)
        logger.info('Number of train images: {:,d}'.format(len(train_dataset)))
    elif phase == 'val':
        val_dataset = ValDataset(dataset_opt)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                num_workers=dataset_opt['workers'], pin_memory=True, collate_fn=my_collate_fn)
        logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_dataset)))
assert train_loader is not None


opt_train = opt['train']
if opt_train['distributed']:
    util.init_distributed_mode(opt)
    num_tasks = util.get_world_size()
    global_rank = util.get_rank()
    sampler_rank = global_rank
    num_training_steps_per_epoch = len(train_dataset) // opt['dataset']['train']['batch_size'] // num_tasks

    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size= opt['dataset']['train']['batch_size'],
        num_workers=opt['dataset']['train']['workers'],
        pin_memory= True,
        prefetch_factor= 4,
        drop_last=True,
        collate_fn=my_collate_fn
    )

model = UnifyDeblur()

def corresponding_load(pre_name, state_dict):
    sub_statedict = {}
    for k, v in state_dict.items():
        if k.startswith(pre_name):
            sub_statedict[k.replace(pre_name, "")] = v
    return sub_statedict

pretrained_ckpt = './pretrained_clipvit_defocus.pth'
pretrained_ckpt = os.path.expanduser(pretrained_ckpt)
checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
print("Load init checkpoint from: %s" % pretrained_ckpt)
checkpoint = checkpoint['state_dict']
checkpoint = corresponding_load('module.', checkpoint)
model.load_state_dict(checkpoint, strict=False)




optimizer = AdamW(model.parameters(), betas=(opt['train']['beta1'], opt['train']['beta2']), weight_decay=1e-2,
                 lr=opt['train']['lr'])


def lambda1(epoch):
    index = 0
    for i,step in enumerate(opt['train']['lr_steps']):
        if epoch < step:
            index = i
            break
    return opt['train']['lr_gamma']**index

def lambda2(epoch):
    return 1

scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                     milestones=opt['train']['lr_steps'],
                                     gamma=opt['train']['lr_gamma'])


# resume training
if resume_state:
    logger.info('Resuming training from epoch: {}.'.format(
        resume_state['epoch']))
    start_epoch = resume_state['epoch']
    optimizer.load_state_dict(resume_state['optimizers'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    scheduler.load_state_dict(resume_state['schedulers'])
    resume_state['state_dict'] = corresponding_load('module.', resume_state['state_dict'])
    model.load_state_dict(resume_state['state_dict'])
else:
    start_epoch = 0

if resume_state:
    optimizer = AdamW(model.parameters(), betas=(opt['train']['beta1'], opt['train']['beta2']), weight_decay=1e-1,
                     lr=opt['train']['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                         milestones=opt['train']['lr_steps'],
                                         gamma=opt['train']['lr_gamma'])
    for _ in range(start_epoch):
        scheduler.step()
        print(scheduler.get_last_lr())

criterion = ReconstructLoss(opt['dataset']['train'])

model = model.cuda()
# training
total_epochs = opt['train']['epoch']

if opt_train['distributed']:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[train_args.local_rank], find_unused_parameters=True)


max_steps = len(train_loader)
logger.info('Start training from epoch: {:d}'.format(start_epoch))
logger.info('# network parameters: {}'.format(sum(param.numel() for param in model.parameters())))

current_step = 0
for epoch in range(start_epoch, total_epochs + 1):
    criterion.iter = epoch
    torch.cuda.empty_cache()
    for index, train_data in tqdm(enumerate(train_loader)):
        gt, b_img, guide_prompts = train_data
        b_img = b_img.cuda()
        gt_img = gt.cuda()
        guide_prompts['img'] = guide_prompts['img'].cuda()

        x = b_img
        blur_parameter, clsblur_embed, clsdeblur_embed, deblur_embed, guidance_embed, deblur_z, guidance_z, \
        recover_img, recover_mae, guidance_mae = model(b_img, gt_img, True)

        losses = criterion(blur_parameter ,clsblur_embed, clsdeblur_embed, deblur_embed, guidance_embed,
                           deblur_z, guidance_z, recover_img, gt_img, None, None, recover_mae, guidance_mae)
        grad_loss = losses["total_loss/final_loss"]
        optimizer.zero_grad()
        grad_loss.backward()
        optimizer.step()
        current_step = epoch * max_steps + index
        # log
        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
            epoch, current_step, scheduler.get_last_lr()[0])
        for k, v in losses.items():
            v = v.cpu().item()
            message += '{:s}: {:.4e} '.format(k, v)
            # tensorboard logger
            if opt['use_tb_logger'] and 'debug' not in opt['name'] and tb_logger is not None:
                tb_logger.add_scalar(k, v, current_step)
        if current_step % opt['logger']['print_freq'] == 0 and train_args.local_rank < 1:
            logger.info(message)

    scheduler.step()

    if epoch % opt['train']['val_freq'] == 0 and train_args.local_rank < 1 :
        torch.cuda.empty_cache()
        avg_psnr = 0.0
        idx = 0
        model.eval()
        for val_data in tqdm(val_loader):
            with torch.no_grad():
                gt, b_img, prompts, root_name = val_data
                gt_img = gt.cuda()
                b_img = b_img.cuda()
                x = b_img
                recover = model(b_img, None, False)
                recover = recover[0]
                # Save ground truth
                img_dir = opt['path']['val_images']
                gt_img_cpu = ToPILImage()(gt_img.squeeze().cpu().clamp(0,1))
                b, g, r = gt_img_cpu.split()
                gt_img_cpu = Image.merge("RGB", (r, g, b))
                recover_img = ToPILImage()(recover.squeeze().cpu().clamp(0,1))
                b, g, r = recover_img.split()
                recover_img = Image.merge("RGB", (r, g, b))
                save_img_path_gt = os.path.join(img_dir, '{:s}_GT_{:d}.jpg'.format(root_name[0][0], current_step))
                gt_img_cpu.save(save_img_path_gt)
                save_img_path_gtr = os.path.join(img_dir,
                                                 "{:s}_recover_{:d}.jpg".format(root_name[0][0], current_step))
                recover_img.save(save_img_path_gtr)

                # calculate psnr
                idx += 1
                avg_psnr += util.calculate_psnr(gt_img, recover)
        model.train()
        avg_psnr = avg_psnr / idx
        # log
        logger.info('# Validation # psnr: {:.4e}.'.format(avg_psnr))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}> psnr: {:.4e}.'.format(epoch, avg_psnr))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            tb_logger.add_scalar('psnr', avg_psnr, current_step)

    # save models and training states
    if epoch % opt['logger']['save_checkpoint_freq'] == 0 and train_args.local_rank < 1:
        logger.info('Saving models and training states.')
        save_filename = '{}_{}.pth'.format(epoch, 'models')
        save_path = os.path.join(opt['path']['models'], save_filename)
        state_dict = model.state_dict()
        save_checkpoint = {'state_dict': state_dict,
                           'optimizers': optimizer.state_dict(),
                           'schedulers': scheduler.state_dict(),
                           'epoch': epoch}
        torch.save(save_checkpoint, save_path)
        torch.cuda.empty_cache()

logger.info('Saving the final model.')
save_filename = 'latest.pth'
save_path = os.path.join(opt['path']['models'], save_filename)
if train_args.local_rank > 0:
    save_checkpoint = {"state_dict": model.state_dict(),
                       'optimizers': optimizer.state_dict(),
                       'schedulers': scheduler.state_dict(),
                       "epoch": opt['train']['epoch']}
    torch.save(save_checkpoint, save_path)
    logger.info('End of training.')
    tb_logger.close()
