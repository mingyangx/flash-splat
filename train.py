#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import torch
from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef
from random import randint
from utils.loss_utils import l1_loss, l1_loss_mask, l2_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# from lpipsPyTorch import lpips
import wandb
import torch.nn as nn
import os
import pickle
import matplotlib.pyplot as plt
import cv2
import shutil

from termcolor import colored   
from random import randint
from utils.xcorr2 import xcorr2   
from utils.isp_torch import render_RAW_image
from utils.myutils import cpuinfo, gpuinfo, info
import piq.psnr as piq_psnr
from render import render_set, render_video

import argparse
import os, shutil
from utils.myutils import seed_torch
from termcolor import colored
from utils.myutils import init_env
from arguments import ModelParams, PipelineParams, OptimizationParams
import sys


class BilateralLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
        self.loss = lambda x: torch.mean(torch.abs(x))
        self.bilateral_filter = lambda x: torch.exp(-torch.abs(x).sum(-1) / self.gamma)
    
    def forward(self, inputs, weights):
        # [1, H, W, C]
        w1 = self.bilateral_filter(weights[:,:,:-1] - weights[:,:,1:])
        w2 = self.bilateral_filter(weights[:,:-1,:] - weights[:,1:,:])
        w3 = self.bilateral_filter(weights[:,:-1,:-1] - weights[:,1:,1:])
        w4 = self.bilateral_filter(weights[:,1:,:-1] - weights[:,:-1,1:])

        L1 = self.loss(w1 * (inputs[:,:,:-1] - inputs[:,:,1:]))
        L2 = self.loss(w2 * (inputs[:,:-1,:] - inputs[:,1:,:]))
        L3 = self.loss(w3 * (inputs[:,:-1,:-1] - inputs[:,1:,1:]))
        L4 = self.loss(w4 * (inputs[:,1:,:-1] - inputs[:,:-1,1:]))
        return (L1 + L2 + L3 + L4) / 4   

class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = lambda x: torch.mean(torch.abs(x))
    
    def forward(self, inputs):
        L1 = self.loss(inputs[:,:,:-1] - inputs[:,:,1:])
        L2 = self.loss(inputs[:,:-1,:] - inputs[:,1:,:])
        L3 = self.loss(inputs[:,:-1,:-1] - inputs[:,1:,1:])
        L4 = self.loss(inputs[:,1:,:-1] - inputs[:,:-1,1:])
        return (L1 + L2 + L3 + L4) / 4 

def training(dataset, opt, pipe, args):
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = args.test_iterations, \
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians_trans_noflash = GaussianModel(args)
    gaussians_trans_flash = GaussianModel(args)
    gaussians_reflect = GaussianModel(args)
    gaussians_beta = GaussianModel(args)

    args.ply_pth_flash = f"{args.source_path}/__ply/15_noflash_fused.ply"
    args.ply_pth_noflash = f"{args.source_path}/__ply/15_flash_fused.ply"
    args.ply_pth_mix = f"{args.source_path}/30_views/dense/fused.ply"
    args.which_scene = 'Transmission'

    scene_trans_noflash = Scene(args, gaussians_trans_noflash, shuffle=False)
    scene_trans_flash = Scene(args, gaussians_trans_flash, shuffle=False)

    viewpoint_stack_trans_noflash = scene_trans_noflash.getTrainCameras().copy()
    for _ in range(len(viewpoint_stack_trans_noflash)):
        viewpoint_cam_trans_noflash = viewpoint_stack_trans_noflash.pop().delete_image()
    viewpoint_stack_trans_noflash = scene_trans_noflash.getTestCameras().copy()
    for _ in range(len(viewpoint_stack_trans_noflash)):
        viewpoint_cam_trans_noflash = viewpoint_stack_trans_noflash.pop().delete_image()

    args.which_scene = 'Reflection'
    scene_reflect = Scene(args, gaussians_reflect, shuffle=False)

    # print(colored('Deleting All Images in Reflect', 'red'))
    viewpoint_stack_reflect = scene_reflect.getTrainCameras().copy()
    for _ in range(len(viewpoint_stack_reflect)):
        viewpoint_cam_reflect = viewpoint_stack_reflect.pop().delete_image()
    viewpoint_stack_reflect = scene_reflect.getTestCameras().copy()
    for _ in range(len(viewpoint_stack_reflect)):
        viewpoint_cam_reflect = viewpoint_stack_reflect.pop().delete_image()

    args.which_scene = 'Beta'
    scene_beta = Scene(args, gaussians_beta, shuffle=False)

    # print(colored('Deleting All Images in Beta', 'red'))
    viewpoint_stack_beta = scene_beta.getTrainCameras().copy()
    for _ in range(len(viewpoint_stack_beta)):
        viewpoint_cam_beta = viewpoint_stack_beta.pop().delete_image()
    viewpoint_stack_beta = scene_beta.getTestCameras().copy()
    for _ in range(len(viewpoint_stack_beta)):
        viewpoint_cam_beta = viewpoint_stack_beta.pop().delete_image()
    # print(colored('Done - Delete All Images in Beta', 'red'))

    # Decide the Idx of Train Cam and Test Cam for Visulizaiton
    viewpoint_stack_trans_flash_train = scene_trans_flash.getTrainCameras()
    viewpoint_stack_trans_flash_test  = scene_trans_flash.getTestCameras()
    # Obtain Flash Noflash Flag for All Cam
    train_cam_list = []
    for i in range(len(viewpoint_stack_trans_flash_train)):
        train_cam_list.append(viewpoint_stack_trans_flash_train[i].check_flash())
    test_cam_list = []
    for i in range(len(viewpoint_stack_trans_flash_test)):
        test_cam_list.append(viewpoint_stack_trans_flash_test[i].check_flash())
    
    vis_test_flash_cam_idx = len(test_cam_list) // 2
    if viewpoint_stack_trans_flash_test[vis_test_flash_cam_idx].check_flash() == False:
        vis_test_noflash_cam_idx = vis_test_flash_cam_idx
        while not viewpoint_stack_trans_flash_test[vis_test_flash_cam_idx].check_flash():
            vis_test_flash_cam_idx += 1
    else:
        vis_test_noflash_cam_idx = vis_test_flash_cam_idx
        while viewpoint_stack_trans_flash_test[vis_test_noflash_cam_idx].check_flash():
            vis_test_noflash_cam_idx += 1

    vis_train_flash_cam_idx = len(train_cam_list) // 2
    if viewpoint_stack_trans_flash_train[vis_train_flash_cam_idx].check_flash() == False:
        vis_train_noflash_cam_idx = vis_train_flash_cam_idx
        while not viewpoint_stack_trans_flash_train[vis_train_flash_cam_idx].check_flash():
            vis_train_flash_cam_idx += 1
    else:
        vis_train_noflash_cam_idx = vis_train_flash_cam_idx
        while viewpoint_stack_trans_flash_train[vis_train_noflash_cam_idx].check_flash():
            vis_train_noflash_cam_idx += 1

    gaussians_trans_noflash.training_setup(opt)
    gaussians_trans_flash.training_setup(opt)
    gaussians_reflect.training_setup(opt)
    gaussians_beta.training_setup(opt)

    # Create Smoothness Loss
    depth_bilateral_loss = BilateralLoss(args.depth_gamma)
    beta_bilateral_loss = BilateralLoss(args.beta_gamma)
    tv_loss = TVLoss()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    viewpoint_stack, pseudo_stack = None, None
    ema_loss_for_log = 0.0
    first_iter += 1

    correlate = xcorr2(zero_mean_normalize=True)

    print(colored('Start Training', 'red'))
    # cpuinfo()
    # gpuinfo()

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene_trans_flash.getTrainCameras().copy()

        # Obtain Camera
        cam_idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(cam_idx)

        # Check if the camera is flash or noflash
        # Obtain Cam Viewpoint
        if args.train_with_paired:
            CAM_IS_FLASH = True if randint(0, 1) == 0 else False
        else:   
            # unpaired data
            CAM_IS_FLASH = viewpoint_cam.check_flash()
        

        # Obtain Rendered Package
        render_pkg_trans_noflash = render(viewpoint_cam, gaussians_trans_noflash, pipe, background)
        render_pkg_trans_flash   = render(viewpoint_cam, gaussians_trans_flash, pipe, background)
        render_pkg_reflect       = render(viewpoint_cam, gaussians_reflect, pipe, background)
        render_pkg_beta          = render(viewpoint_cam, gaussians_beta, pipe, background)
        
        # Obtain Trans_noflash/Trans_flash/Reflect Image 
        image_trans_noflash, viewspace_point_tensor_trans_noflash, visibility_filter_trans_noflash, radii_trans_noflash = render_pkg_trans_noflash["render"], render_pkg_trans_noflash["viewspace_points"], render_pkg_trans_noflash["visibility_filter"], render_pkg_trans_noflash["radii"]
        image_trans_flash, viewspace_point_tensor_trans_flash, visibility_filter_trans_flash, radii_trans_flash = render_pkg_trans_flash["render"], render_pkg_trans_flash["viewspace_points"], render_pkg_trans_flash["visibility_filter"], render_pkg_trans_flash["radii"]
        image_reflect, viewspace_point_tensor_reflect, visibility_filter_reflect, radii_reflect = render_pkg_reflect["render"], render_pkg_reflect["viewspace_points"], render_pkg_reflect["visibility_filter"], render_pkg_reflect["radii"]
        
        viewspace_point_tensor_beta, visibility_filter_beta, radii_beta = render_pkg_beta["viewspace_points"], render_pkg_beta["visibility_filter"], render_pkg_beta["radii"]

        # Obtain Beta
        beta = render_pkg_beta["render"][0]
        # beta = torch.clamp(beta, 0.0, 1.0)
        beta = torch.sigmoid(beta)
        beta_clone = beta.clone()

        # render RAW
        if args.use_raw:
            image_trans_flash = image_trans_flash ** (1/args.gamma_corr)  # rgb to raw
            image_trans_noflash = image_trans_noflash ** (1/args.gamma_corr)
            image_reflect = image_reflect ** (1/args.gamma_corr)   
                
        num_pc_reflect = gaussians_reflect.get_features.size()[0]
        num_pc_trans_noflash = gaussians_trans_noflash.get_features.size()[0]   
        num_pc_trans_flash = gaussians_trans_flash.get_features.size()[0]
        num_pc_beta = gaussians_beta.get_features.size()[0]

        num_xyz_reflect = len(gaussians_reflect._xyz)
        num_xyz_trans_noflash = len(gaussians_trans_noflash._xyz)
        num_xyz_trans_flash = len(gaussians_trans_flash._xyz)
        num_xyz_beta = len(gaussians_beta._xyz)

        # take the max of the  num_xyz_reflect, num_xyz_trans_noflash, num_xyz_trans_flash, num_xyz_beta
        num_xyz_list = [num_xyz_reflect, num_xyz_trans_flash, num_xyz_trans_noflash, num_xyz_beta]
        sum_num_xyz = sum(num_xyz_list)
        max_num_xyz = max(num_xyz_list)

        freeze_reflect, freeze_trans_flash, freeze_trans_noflash, freeze_beta = False, False, False, False
        
        if sum_num_xyz > 3e6:
            freeze_reflect = num_xyz_reflect == max_num_xyz
            freeze_trans_flash = num_xyz_trans_flash == max_num_xyz
            freeze_trans_noflash = num_xyz_trans_noflash == max_num_xyz
            freeze_beta = num_xyz_beta == max_num_xyz
        
        # log the number of points to wandb
        if args.use_wandb:
            wandb.log({'num_pc_reflect': num_pc_reflect, 
                    'num_pc_trans_noflash': num_pc_trans_noflash, 
                    'num_pc_trans_flash': num_pc_trans_flash, 
                    'num_pc_beta': num_pc_beta}, step=iteration)

            wandb.log({'num_xyz_reflect': num_xyz_reflect, 
                    'num_xyz_trans_noflash': num_xyz_trans_noflash, 
                    'num_xyz_trans_flash': num_xyz_trans_flash, 
                    'num_xyz_beta': num_xyz_beta}, step=iteration)      

        image_flash   = image_trans_flash   + image_reflect * beta
        image_noflash = image_trans_noflash + image_reflect * beta

        if args.train_with_paired:
            gt_image = viewpoint_cam.original_image if CAM_IS_FLASH else viewpoint_cam.original_image_noflash
            image = image_flash if CAM_IS_FLASH else image_noflash
        else:
            gt_image = viewpoint_cam.original_image
            image = image_flash if CAM_IS_FLASH else image_noflash

        if args.use_raw:
            if not args.loss_at_raw:
                gt_image = gt_image ** args.gamma_corr
                image = image ** args.gamma_corr

        # RGB Loss
        Ll1 =  l1_loss_mask(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        loss *= args.rgb_weight
        if args.use_wandb:
            wandb.log({'L1 and SSIM': loss.item()}, step=iteration)


        psnr_metric = piq_psnr(torch.clamp(render_RAW_image(image), 0, 1).unsqueeze(0), torch.clamp(render_RAW_image(gt_image), 0, 1).unsqueeze(0))
        if args.use_wandb:
            wandb.log({'PSNR': psnr_metric}, step=iteration)
    
        # flash/noflash Loss
        scaled_trans_noflash = image_trans_noflash / image_trans_noflash.mean() * image_trans_flash.mean()
        if args.fnf_mode == 'pearson':
            fnf_loss =  - pearson_corrcoef(image_trans_noflash.flatten(), image_trans_flash.flatten())
        elif args.fnf_mode == 'mse':
            fnf_loss = ((scaled_trans_noflash - image_trans_flash) ** 2).mean()  
        elif args.fnf_mode == 'ncc': # translation invariante
            fnf_loss = correlate(image_trans_flash.unsqueeze(0), scaled_trans_noflash.unsqueeze(0)).squeeze()
        elif args.fnf_mode == 'stlpips': # translation invariant net
            pass
            # fnf_loss = stlpips_metric(image_trans_flash.unsqueeze(0), scaled_trans_noflash.unsqueeze(0)).squeeze()
        else:
            raise ValueError(f'fnf_mode must be one of pearson, mse, ncc, stlpips, but got {args.fnf_mode}')
        loss += args.fnf_weight * fnf_loss

        if args.use_wandb:
            wandb.log({'Flash NoFlash': (args.fnf_weight * fnf_loss).item()}, step=iteration)
        # Beta Smoothness Loss
        if args.beta_type == 'tv':
            beta_loss = tv_loss(beta.unsqueeze(0))
        elif args.beta_type == 'bilateral':
            beta_loss = beta_bilateral_loss(beta_clone.unsqueeze(0), beta_clone.unsqueeze(0).unsqueeze(-1))
        elif args.beta_type == '' or args.beta_type == None or args.beta_type.lower() == 'none':
            beta_loss = 0

        else:
            raise ValueError('args.beta_type issue')
        loss += args.beta_weight * beta_loss

        if args.use_wandb:
            wandb.log({'Beta Smoothness': (args.beta_weight * beta_loss).item()}, step=iteration)

        # Reflect DS Loss
        if args.depth_type == 'bilateral':
            reflect_depth_loss = depth_bilateral_loss(render_pkg_reflect["depth"][0].unsqueeze(0), render_pkg_reflect["depth"][0].unsqueeze(0).unsqueeze(-1))
            trans_flash_depth_loss = depth_bilateral_loss(render_pkg_trans_flash["depth"][0].unsqueeze(0), render_pkg_trans_flash["depth"][0].unsqueeze(0).unsqueeze(-1))
            trans_noflash_depth_loss = depth_bilateral_loss(render_pkg_trans_noflash["depth"][0].unsqueeze(0), render_pkg_trans_noflash["depth"][0].unsqueeze(0).unsqueeze(-1))
        elif args.depth_type == '' or args.depth_type == None or args.depth_type.lower() == 'none':
            reflect_depth_loss = 0
            trans_depth_loss = 0
        else:
            raise ValueError('args.depth_type issue')

        loss += args.reflect_depth_weight * reflect_depth_loss
        loss += args.trans_depth_weight * trans_flash_depth_loss
        loss += args.trans_depth_weight * trans_noflash_depth_loss

        if args.use_wandb:
            wandb.log({'Depth Smoothness': (args.reflect_depth_weight * reflect_depth_loss + args.trans_depth_weight * trans_flash_depth_loss + args.trans_depth_weight * trans_noflash_depth_loss).item()}, 
                  step=iteration)

        loss.backward()

        if 'debug' in args.vis.lower():
            if iteration % 10 == 0:
                cpuinfo(f'Debug Iteration {iteration} CPU')
                gpuinfo(f'Debug Iteration {iteration} GPU', use_wandb=args.use_wandb, wandb_iter=iteration)
    
        if iteration % 10 == 0 and args.use_wandb:
            gpuinfo(f'Debug Iteration {iteration} GPU', use_wandb=args.use_wandb, wandb_iter=iteration)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in testing_iterations:

                # Obtain All Cam Viewpoints
                all_view_points = scene_trans_noflash.getTestCameras() + scene_trans_noflash.getTrainCameras()
                
                # Render GIF
                render_video(args.source_path, args.model_path, scene_trans_noflash.loaded_iter, scene_trans_noflash.getTrainCameras(), [gaussians_reflect, gaussians_beta, gaussians_trans_flash, gaussians_trans_noflash], pipe, background, 25, demo_gif_path=args.demo_gif_path, exp_name=args.exp_name, gamma_corr=args.gamma_corr)

                # Render Test and Train View Seperately
                render_set(args.model_path, 'train_view', scene_trans_noflash.loaded_iter, scene_trans_noflash.getTrainCameras(), [gaussians_reflect, gaussians_beta, gaussians_trans_flash, gaussians_trans_noflash], pipe, background, 25)
                render_set(args.model_path, 'test_view', scene_trans_noflash.loaded_iter, scene_trans_noflash.getTestCameras(), [gaussians_reflect, gaussians_beta, gaussians_trans_flash, gaussians_trans_noflash], pipe, background, 25)

                base_fig_num = 0
                fig, axes = None, None
                for train_test_choice in ['train', 'test']:
                    for fnf_choice in ['flash', 'noflash']:
                        # Choose Current Vis Cam
                        if train_test_choice == 'train' and fnf_choice == 'flash':
                            vis_cam_idx = vis_train_flash_cam_idx
                        elif train_test_choice == 'train' and fnf_choice == 'noflash':
                            vis_cam_idx = vis_train_noflash_cam_idx
                        elif train_test_choice == 'test' and fnf_choice == 'flash':
                            vis_cam_idx = vis_test_flash_cam_idx
                        elif train_test_choice == 'test' and fnf_choice == 'noflash':
                            vis_cam_idx = vis_test_noflash_cam_idx
                        else:
                            raise ValueError('train_test_choice or fnf_choice issue')

                        # Log and Save
                        trans_flash_image, gt_image, gt_midas_depth, trans_flash_depth = training_report_trans_flash(tb_writer, iteration, Ll1, loss, l1_loss,
                                        testing_iterations, scene_trans_flash, render, (pipe, background), train_test_choice, vis_cam_idx, args)
                        trans_noflash_image, trans_noflash_depth = training_report_trans_noflash(tb_writer, iteration, Ll1, loss, l1_loss,
                                        testing_iterations, scene_trans_noflash, render, (pipe, background), train_test_choice, vis_cam_idx, args)
                    
                        reflect_image, reflect_depth = training_report_reflect(tb_writer, iteration, Ll1, loss, l1_loss,
                                        testing_iterations, scene_reflect, render, (pipe, background), train_test_choice, vis_cam_idx, args)

                        beta = training_report_beta(tb_writer, iteration, Ll1, loss, l1_loss,
                                        testing_iterations, scene_beta, render, (pipe, background), train_test_choice, vis_cam_idx, args)

                        fig, axes = plot_all_image(iteration, gt_image, gt_midas_depth, trans_flash_image, trans_noflash_image, trans_flash_depth, trans_noflash_depth, reflect_image, reflect_depth, beta, args.compare_pth, args.exp_name, fnf_choice, train_test_choice, base_fig_num, fig, axes, args)    
                        
                        base_fig_num += 2
            
            if iteration > first_iter and (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene_trans_flash.save(iteration)

            if iteration > first_iter and (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                        scene_trans_flash.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Densification
            if  iteration < opt.densify_until_iter:
                args.splat_ceiling = 0.5e6
                if not freeze_trans_flash and num_xyz_trans_flash < args.splat_ceiling:
                    # Keep track of max radii in image-space for pruning
                    gaussians_trans_flash.max_radii2D[visibility_filter_trans_flash] = torch.max(gaussians_trans_flash.max_radii2D[visibility_filter_trans_flash], radii_trans_flash[visibility_filter_trans_flash])
                    gaussians_trans_flash.add_densification_stats(viewspace_point_tensor_trans_flash, visibility_filter_trans_flash, freeze=freeze_trans_flash)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = None
                        gaussians_trans_flash.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene_trans_flash.cameras_extent, size_threshold, iteration, freeze=freeze_trans_flash)


                if not freeze_trans_noflash and num_xyz_trans_noflash < args.splat_ceiling:
                    gaussians_trans_noflash.max_radii2D[visibility_filter_trans_noflash] = torch.max(gaussians_trans_noflash.max_radii2D[visibility_filter_trans_noflash], radii_trans_noflash[visibility_filter_trans_noflash])
                    gaussians_trans_noflash.add_densification_stats(viewspace_point_tensor_trans_noflash, visibility_filter_trans_noflash, freeze=freeze_trans_noflash)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = None
                        gaussians_trans_noflash.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene_trans_noflash.cameras_extent, size_threshold, iteration, freeze=freeze_trans_noflash)
                    
                if not freeze_reflect and num_xyz_reflect < args.splat_ceiling:
                    gaussians_reflect.max_radii2D[visibility_filter_reflect] = torch.max(gaussians_reflect.max_radii2D[visibility_filter_reflect], radii_reflect[visibility_filter_reflect])
                    gaussians_reflect.add_densification_stats(viewspace_point_tensor_reflect, visibility_filter_reflect, freeze=freeze_reflect)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = None
                        gaussians_reflect.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene_reflect.cameras_extent, size_threshold, iteration, freeze=freeze_reflect)

                if args.beta_enabled and not freeze_beta and num_xyz_beta < args.splat_ceiling:
                    gaussians_beta.max_radii2D[visibility_filter_beta] = torch.max(gaussians_beta.max_radii2D[visibility_filter_beta], radii_beta[visibility_filter_beta])
                    gaussians_beta.add_densification_stats(viewspace_point_tensor_beta, visibility_filter_beta, freeze=freeze_beta)
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = None
                        gaussians_beta.densify_and_prune(opt.densify_grad_threshold, opt.prune_threshold, scene_beta.cameras_extent, size_threshold, iteration, freeze=freeze_beta)

            # Optimizer step
            if iteration < opt.iterations:
                if CAM_IS_FLASH:
                    gaussians_trans_flash.optimizer.step()
                    gaussians_trans_flash.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians_trans_noflash.optimizer.step()
                    gaussians_trans_noflash.optimizer.zero_grad(set_to_none = True)
                gaussians_reflect.optimizer.step()
                gaussians_reflect.optimizer.zero_grad(set_to_none = True)
                if args.beta_enabled:
                    gaussians_beta.optimizer.step()
                    gaussians_beta.optimizer.zero_grad(set_to_none = True)

            gaussians_trans_flash.update_learning_rate(iteration)
            gaussians_trans_noflash.update_learning_rate(iteration)
            gaussians_reflect.update_learning_rate(iteration)
            if args.beta_enabled:
                gaussians_beta.update_learning_rate(iteration)
            if (iteration - args.start_sample_pseudo - 1) % opt.opacity_reset_interval == 0 and \
                    iteration > args.start_sample_pseudo:
                gaussians_trans_flash.reset_opacity()
                gaussians_trans_noflash.reset_opacity()
                gaussians_reflect.reset_opacity()
                if args.beta_enabled:
                    gaussians_beta.reset_opacity()

    torch.cuda.empty_cache()    

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True, mode = 0o777)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        pass
        # print("Tensorboard not available: not logging progress")

    return tb_writer

def training_report_trans_flash(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_choice, vis_cam_idx, args=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        if train_test_choice == 'test':
            config = validation_configs[0]
        else:
            config = validation_configs[1]
        

        if config['cameras'] and len(config['cameras']) > 0:
            l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0

            viewpoint = config['cameras'][vis_cam_idx]

            image, depth = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], renderFunc(viewpoint, scene.gaussians, *renderArgs)["depth"][0]
            gt_image = viewpoint.original_image.to("cuda")

            midas_depth = torch.zeros((400, 400))
            # normed depth
            midas_depth = -midas_depth
            midas_depth = (midas_depth - torch.min(midas_depth)) / (torch.max(midas_depth) - torch.min(midas_depth))
            # Inverse depth
            depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))

            return image, gt_image, midas_depth, depth

    return None, None, None, None

def training_report_trans_noflash(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_choice, vis_cam_idx, args=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        if train_test_choice == 'test':
            config = validation_configs[0]
        else:
            config = validation_configs[1]
        
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
            viewpoint = config['cameras'][vis_cam_idx]
            image, depth = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], renderFunc(viewpoint, scene.gaussians, *renderArgs)["depth"][0]
            # image = torch.clamp(image, 0.0, 1.0)
            depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))

            return image, depth

    return None, None


def training_report_reflect(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_choice, vis_cam_idx, args):

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        if train_test_choice == 'test':
            config = validation_configs[0]
        else:
            config = validation_configs[1]

        if config['cameras'] and len(config['cameras']) > 0:
            l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
            viewpoint = config['cameras'][vis_cam_idx]
            image, depth = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], renderFunc(viewpoint, scene.gaussians, *renderArgs)["depth"][0]
            


            # Inverse depth
            depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))

            return image, depth

    return None, None

def training_report_beta(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_choice, vis_cam_idx, args):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        if train_test_choice == 'test':
            config = validation_configs[0]
        else:
            config = validation_configs[1]

        if config['cameras'] and len(config['cameras']) > 0:
            l1_test, psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0, 0.0
            viewpoint = config['cameras'][vis_cam_idx]
            beta = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"][0]
            # beta = torch.clamp(beta, 0.0, 1.0)
            beta = torch.sigmoid(beta)

            return beta

    return None, None


def normalize(x):
    x =  (x - x.min()) / (x.max() - x.min())
    return x


def plot_all_image(iteration, gt_image, gt_midas_depth, trans_flash_image, trans_noflash_image, trans_flash_depth, trans_noflash_depth, reflect_image, reflect_depth, beta, compare_pth, expname, fnf_choice, train_test_choice, base_fig_num, fig, axs, args):

        reflect_beta = reflect_image * beta
        combine_flash_img = trans_flash_image + reflect_beta
        combine_noflash_img = trans_noflash_image + reflect_beta

        if args.use_raw:
            raw_dict = {
                'gt_image': gt_image,
                'trans_flash_image': trans_flash_image,
                'trans_noflash_image': trans_noflash_image,
                'reflect_image': reflect_image,
                'gamma_corr': args.gamma_corr,
            }
            
            # save the dictionary
            if args.save_data_during_training:
                with open(compare_pth + f'/{expname}_{iteration}_{train_test_choice}_{fnf_choice}_raw.pkl', 'wb') as f:
                    pickle.dump(raw_dict, f)

            gt_image = render_RAW_image(gt_image)

            if args.loss_at_raw: # trained in raw space
                trans_flash_image = render_RAW_image(trans_flash_image ** (1/args.gamma_corr))
                trans_noflash_image = render_RAW_image(trans_noflash_image ** (1/args.gamma_corr))
                reflect_image = render_RAW_image(reflect_image ** (1/args.gamma_corr))  

                reflect_beta = reflect_image * beta
                combine_flash_img = trans_flash_image + reflect_beta
                combine_noflash_img = trans_noflash_image + reflect_beta
            else:
                trans_flash_ratio = gt_image.mean() / trans_flash_image.mean()
                trans_noflash_ratio = gt_image.mean() / trans_noflash_image.mean()
                reflect_ratio = gt_image.mean() / reflect_image.mean()

                reflect_beta = reflect_image ** (1/args.gamma_corr) * beta
                combine_flash_img = render_RAW_image(trans_flash_image ** (1/args.gamma_corr) + reflect_beta)
                combine_noflash_img = render_RAW_image(trans_noflash_image ** (1/args.gamma_corr) + reflect_beta)

                trans_flash_image = render_RAW_image(trans_flash_image ** (1/args.gamma_corr) * trans_flash_ratio)
                trans_noflash_image = render_RAW_image(trans_noflash_image ** (1/args.gamma_corr) * trans_noflash_ratio)
                reflect_image = render_RAW_image(reflect_image ** (1/args.gamma_corr) * reflect_ratio)  

                reflect_beta_ratio = gt_image.mean() / reflect_beta.mean()
                reflect_beta = render_RAW_image(reflect_beta * reflect_beta_ratio)

                # make the max pixel 1
        

        trans_flash_image, trans_noflash_image, gt_image, reflect_image = trans_flash_image.permute(1, 2, 0), trans_noflash_image.permute(1, 2, 0) ,gt_image.permute(1, 2, 0), reflect_image.permute(1, 2, 0)
        trans_flash_image, trans_noflash_image, gt_image, reflect_image = trans_flash_image.cpu().numpy(), trans_noflash_image.cpu().numpy(), gt_image.cpu().numpy(), reflect_image.cpu().numpy()
        
        reflect_beta, combine_flash_img, combine_noflash_img = reflect_beta.permute(1, 2, 0), combine_flash_img.permute(1, 2, 0), combine_noflash_img.permute(1, 2, 0)
        reflect_beta, combine_flash_img, combine_noflash_img = reflect_beta.cpu().numpy(), combine_flash_img.cpu().numpy(), combine_noflash_img.cpu().numpy()

        # matplot all depth and beta
        gt_midas_depth, trans_flash_depth, trans_noflash_depth, reflect_depth, beta = gt_midas_depth.cpu().numpy(), trans_flash_depth.cpu().numpy(), trans_noflash_depth.cpu().numpy(), reflect_depth.cpu().numpy(), beta.cpu().numpy()

        '''
        create a dictionary to save all the images and depth
        gt_image, trans_flash_image, trans_noflash_image, trans_flash_depth, trans_noflash_depth, reflect_image, reflect_depth, beta
        reflect_beta, combine_flash_img, combine_noflash_img
        '''
        save_dict = {
            'gt_image': gt_image,
            'trans_flash_image': trans_flash_image,
            'trans_noflash_image': trans_noflash_image,
            'reflect_image': reflect_image,
            'reflect_beta': reflect_beta,
            'combine_flash_img': combine_flash_img,
            'combine_noflash_img': combine_noflash_img,
            'trans_flash_depth': trans_flash_depth,
            'trans_noflash_depth': trans_noflash_depth,
            'reflect_depth': reflect_depth,
            'beta': beta
            }
        
        if args.save_data_during_training:
            # save the dictionary 
            with open(compare_pth + f'/{expname}_{iteration}_{train_test_choice}_{fnf_choice}.pkl', 'wb') as f:
                pickle.dump(save_dict, f)

        # clamp trans_image, refect_image, reflect_beta, combine_img
        gt_image = np.clip(gt_image, 0, 1)
        trans_flash_image = np.clip(trans_flash_image, 0, 1)
        trans_noflash_image = np.clip(trans_noflash_image, 0, 1)
        reflect_image = np.clip(reflect_image, 0, 1)
        reflect_beta = np.clip(reflect_beta, 0, 1)
        combine_flash_img = np.clip(combine_flash_img, 0, 1)
        combine_noflash_img = np.clip(combine_noflash_img, 0, 1)
        

        trans_flash_image, trans_noflash_image, gt_image, reflect_image, reflect_beta = (trans_flash_image * 255).astype('uint8'), (trans_noflash_image * 255).astype('uint8'), (gt_image * 255).astype('uint8'), (reflect_image * 255).astype('uint8'), (reflect_beta * 255).astype('uint8')
        combine_flash_img = (combine_flash_img * 255).astype('uint8')
        combine_noflash_img = (combine_noflash_img * 255).astype('uint8')
        
        # 2x5 subfigures.
        if base_fig_num == 0:
            fig, axs = plt.subplots(2*4, 7, figsize=(24, 24))

        row_1 = base_fig_num 
        row_2 = base_fig_num + 1
        # First Row
        im1 = axs[row_1, 0].matshow(gt_midas_depth, cmap='viridis')
        axs[row_1, 0].set_title('gt_midas_depth')

        # Blank
        axs[row_1, 1].axis('off')

        im2 = axs[row_1, 2].matshow(trans_flash_depth, cmap='viridis')
        axs[row_1, 2].set_title(f"{train_test_choice}_flash_Trans")

        axs[row_1, 3].axis('off')

        img3 = axs[row_1, 4].matshow(trans_noflash_depth, cmap='viridis')
        axs[row_1, 4].set_title(f"{train_test_choice}_noflash_Trans")

        img4 = axs[row_1, 5].matshow(reflect_depth, cmap='viridis')
        axs[row_1, 5].set_title(f"{train_test_choice}_Reflect")

        # beta hist
        axs[row_1, 6].matshow(beta, cmap='viridis')
        axs[row_1, 6].set_title('beta hist')

        # Second Row (For Image Plots)
        axs[row_2, 0].imshow(gt_image)
        axs[row_2, 0].set_title(f'{train_test_choice}_{fnf_choice}_GT')

        axs[row_2, 1].imshow(combine_flash_img)
        axs[row_2, 1].set_title(f"{train_test_choice}_flash_Combined")

        axs[row_2, 2].imshow(trans_flash_image)
        axs[row_2, 2].set_title(f"{train_test_choice}_flash_Trans")

        axs[row_2, 3].imshow(combine_noflash_img)
        axs[row_2, 3].set_title(f"{train_test_choice}_noflash_Combined")

        axs[row_2, 4].imshow(trans_noflash_image)
        axs[row_2, 4].set_title(f"{train_test_choice}_noflash_Trans")

        axs[row_2, 5].imshow(reflect_image)
        axs[row_2, 5].set_title(f"{train_test_choice}_Reflect")

        # Placeholder for the fifth column in the second row
        axs[row_2, 6].imshow(reflect_beta)
        axs[row_2, 6].set_title(f"{train_test_choice}_Reflect_beta")

        if base_fig_num == 6:
            fig.suptitle('{}_{}'.format(expname, iteration))
            plt.tight_layout()  # Adjust layout to prevent overlap
            plt.savefig(compare_pth + '/{}_{}.png'.format(expname, iteration))
            plt.close()
            if args.use_wandb:
                wandb.log({"Recon": wandb.Image(compare_pth + '/{}_{}.png'.format(expname, iteration))})

        else:
            return fig, axs
        return None, None

def nerflash(
    param_dict,
    vis = '../demo', 
    exp_name = None,
    ours=True,
    beta_type='bilateral', # midas | smooth | both
    depth_type='bilateral',
    train_with_paired = False, # 'paired' or 'unpaired'  
    fnf_mode = 'pearson', # pearson | mse | ncc | stlpips
    use_raw = True,
    gamma_corr = 0.22,
    loss_at_raw=False,
    save_data_during_training = False,
):
    init_env()
    seed_torch(0)
    parser = argparse.ArgumentParser(description='Run Flash-Splat')
    parser.add_argument('--exp_name', type=str, default='Default_Run', help='Save folder')
    parser.add_argument('--data_dir', type=str, default='/fs/nexus-projects/imaging_through_aberration/nerflash/Unpaired_Flash_NoFlash_Dataset', help='data directory')
    parser.add_argument('--save_dir', type=str, default='experiments', help='save directory')
    parser.add_argument('--gif_save_dir', type=str, default='demo', help='save directory for GIFs')
    parser.add_argument('--turn_on_wandb', action='store_true', help='use wandb') # default is False

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    sys.argv += ['--eval']
    args = parser.parse_args(sys.argv[1:])

    scene_name = param_dict['scene_name']
    beta_g_exp = param_dict['beta_gamma']
    beta_w_exp = param_dict['beta_weight']
    depth_g_exp = param_dict['depth_gamma']
    tran_w_exp = param_dict['depth_weight']
    refl_w_exp = param_dict['depth_weight']
    fnf_weight_exp = param_dict['fnf_weight_exp']
    rgb_weight_exp = param_dict['rgb_weight_exp']
    test_iterations = param_dict['test_iterations']
    max_iter = param_dict['iterations']
    n_views = param_dict['n_views']

    args.ip = '127.0.0.1'
    args.port = 6009
    args.debug_from = -1
    args.detect_anomaly = False
    args.test_iterations = test_iterations
    args.save_iterations = [1000_00]
    args.quiet = False
    args.checkpoint_iterations = [1000_00]
    args.start_checkpoint = None
    args.train_bg = False
    args.use_wandb = args.turn_on_wandb
    source_path = f'{args.data_dir}/{scene_name}'
    args.source_path = source_path
    args.n_views = n_views

    demo_gif_path = f'{args.gif_save_dir}/{args.exp_name}'
    exp_folder = f'{args.save_dir}/{args.exp_name}'
    if 'debug' in exp_folder:
        shutil.rmtree(exp_folder, ignore_errors=True)
    print(colored(exp_folder, 'red'))
    for folder in [exp_folder, demo_gif_path]:
        os.makedirs(folder, exist_ok=True)

    paired_name = 'paired' if train_with_paired else 'unpaired'
    raw_name = f'rgb_exp{rgb_weight_exp}' if use_raw else 'png'
    vis = exp_folder

    args.vis = exp_folder
    args.ours = ours
    args.quiet = False
    args.beta_gamma = 10 ** beta_g_exp
    args.beta_weight = 10 ** beta_w_exp
    args.depth_gamma = 10 ** depth_g_exp
    args.trans_depth_weight = 10 ** tran_w_exp 
    args.reflect_depth_weight = 10 ** refl_w_exp
    args.fnf_weight = 10 ** fnf_weight_exp
    args.rgb_weight = 10 ** rgb_weight_exp
    args.gamma_corr = gamma_corr    
    args.beta_type    = beta_type # midas | smooth | both
    args.depth_type = depth_type
    args.beta_enabled = True
    args.ori_depth_loss_enable = False  # whether ori depth loss (midas)
    args.use_raw = use_raw
    args.save_data_during_training = save_data_during_training
    op.iterations = max_iter
    args.iterations = max_iter
    args.demo_gif_path = demo_gif_path
    args.save_iterations = [1000_00]
    args.save_iterations.append(args.iterations)
    args.test_iterations = test_iterations
    args.loss_at_raw = loss_at_raw

    curr_date = datetime.datetime.now().strftime("%m%d")
    vis_child_folder = vis.split('/')[-1]

    args.exp_name = scene_name + "_" + vis_child_folder

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # get last part of the vis directory
    args.model_path = f'{exp_folder}/{args.exp_name}_{timestamp}'
    args.compare_pth = args.model_path
    os.makedirs(args.compare_pth, exist_ok=True, mode=0o777)

    args.train_with_paired = train_with_paired # 'paired' or 'unpaired'
    args.fnf_mode = fnf_mode # pearson | mse | ncc | stlpips    

    # print(args.test_iterations)
    print("Optimizing " + args.model_path)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    if args.ours:
        if args.use_wandb:
            run = wandb.init(project="Flash_n_Splat", name=f'{args.exp_name}', reinit=True, dir=f'{exp_folder}/wandb', settings=wandb.Settings(start_method="fork"))
        try:
            training(lp.extract(args), op.extract(args), pp.extract(args), args)
        # catch error message and print it
        except Exception as e: 
            print(colored(e, 'red'))
            print(colored(args.exp_name, 'red'))    
            print(colored(args.vis, 'red'))
            if args.use_wandb:
                wandb.alert(title="Error", text=f"Error in {exp_name}: {e}", level="error")
            # print current time
            print(colored(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), 'red')
            if args.use_wandb:
                run.finish()
            exit()
    else:
        raise ValueError('Please Check Which Model You Want to Train. NeRFReN or FSGS')

    # All done
    print("\nTraining complete.")
    if args.use_wandb:
        run.finish()
    print(colored(f'Run {args.vis} {args.exp_name} finished', 'red'))
