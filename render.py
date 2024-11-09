#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import matplotlib.pyplot as plt
import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import time
from tqdm import tqdm

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
from utils.general_utils import vis_depth
from utils.isp_torch import render_RAW_image
from utils.myutils import info, normalize
from termcolor import colored
import shutil


# def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

#     makedirs(render_path, exist_ok=True, mode=0o777)
#     makedirs(gts_path, exist_ok=True, mode=0o777)

#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         rendering = render(view, gaussians, pipeline, background)
#         gt = view.original_image[0:3, :, :]
#         torchvision.utils.save_image(rendering["render"], os.path.join(render_path, view.image_name + '.png'))
#                                             #'{0:05d}'.format(idx) + ".png"))
#         torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

#         if args.render_depth:
#             depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
#             np.save(os.path.join(render_path, view.image_name + '_depth.npy'), rendering['depth'][0].detach().cpu().numpy())
#             cv2.imwrite(os.path.join(render_path, view.image_name + '_depth.png'), depth_map)


def render_set(model_path, name, iteration, views, gaussians_list, pipeline, background, args):
    # mkdir
    name_list = ['refl', 'beta', 'refl_beta', 'flash_trans', 'flash_comp', 'noflash_trans', 'noflash_comp']
    render_folder = []
    for name in name_list:
        render_path = os.path.join(model_path, 'all_viewpoints_test', f"{name}")
        # remove older folder if exists. Create a new one.
        if os.path.exists(render_path):
            shutil.rmtree(render_path)
        makedirs(render_path, exist_ok=True, mode=0o777)
        render_folder.append(render_path)

    gs_refl = gaussians_list[0]
    gs_beta = gaussians_list[1]
    gs_trans_flash = gaussians_list[2]
    gs_trans_noflash = gaussians_list[3]


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        rendering_refl = torch.clamp(render(view, gs_refl, pipeline, background)["render"], min=0., max=1.)
        rendering_beta = torch.clamp(render(view, gs_beta, pipeline, background)["render"], min=0., max=1.)
        rendering_trans_flash = torch.clamp(render(view, gs_trans_flash, pipeline, background)["render"], min=0., max=1.)
        rendering_trans_noflash = torch.clamp(render(view, gs_trans_noflash, pipeline, background)["render"], min=0., max=1.)

        torchvision.utils.save_image(rendering_refl, os.path.join(render_folder[0], view.image_name + ".png"))
        torchvision.utils.save_image(rendering_beta, os.path.join(render_folder[1], view.image_name + ".png"))
        torchvision.utils.save_image(rendering_refl * rendering_beta, os.path.join(render_folder[2], view.image_name + ".png"))
        torchvision.utils.save_image(rendering_trans_flash, os.path.join(render_folder[3], view.image_name + ".png"))
        torchvision.utils.save_image(rendering_trans_flash + rendering_refl * rendering_beta, os.path.join(render_folder[4], view.image_name + ".png"))
        torchvision.utils.save_image(rendering_trans_noflash, os.path.join(render_folder[5], view.image_name + ".png"))
        torchvision.utils.save_image(rendering_trans_noflash + rendering_refl * rendering_beta, os.path.join(render_folder[6], view.image_name + ".png"))



def render_video(source_path, model_path, iteration, views, gaussians_list, pipeline, background, name = 'None', fps=30, demo_gif_path=None, exp_name=None, gamma_corr=0.22):
    
    # mkdir
    name_list = ['refl', 'beta', 'refl_beta', 'flash_trans', 'flash_comp', 'noflash_trans', 'noflash_comp']
    render_folder = []
    for name in name_list:
        render_path = os.path.join(model_path, 'video', f"{name}")
        # remove older folder if exists. Create a new one.
        if os.path.exists(render_path):
            shutil.rmtree(render_path)
        makedirs(render_path, exist_ok=True, mode=0o777)
        render_folder.append(render_path)


    gs_refl = gaussians_list[0]
    gs_beta = gaussians_list[1]

    gs_trans_flash = gaussians_list[2]
    gs_trans_noflash = gaussians_list[3]

    

    view = views[0]
    render_poses = generate_spiral_path(np.load(source_path + '/poses_bounds.npy'))
    img_list = []
    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        # rendering = render(view, gaussians, pipeline, background)
        rendering_refl = render(view, gs_refl, pipeline, background)["render"]
        rendering_beta = render(view, gs_beta, pipeline, background)["render"]
        rendering_trans_flash = render(view, gs_trans_flash, pipeline, background)["render"]
        rendering_trans_noflash = render(view, gs_trans_noflash, pipeline, background)["render"]

        rendering_beta = torch.sigmoid(rendering_beta)


        # torchvision.utils.save_image(rendering_refl, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(torch.clamp(rendering_refl, min=0., max=1.), render_folder[0] + '/{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(torch.clamp(rendering_beta, min=0., max=1.), render_folder[1] + '/{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering_refl * rendering_beta, render_folder[2] + '/{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(torch.clamp(rendering_trans_flash, min=0., max=1.), render_folder[3] + '/{0:05d}'.format(idx) + ".png")
        # torchvision.utils.save_image(rendering_trans_flash + rendering_refl * rendering_beta, render_folder[4] + '/{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(torch.clamp(rendering_trans_noflash, min=0., max=1.), render_folder[5] + '/{0:05d}'.format(idx) + ".png")
        # torchvision.utils.save_image(rendering_trans_noflash + rendering_refl * rendering_beta, render_folder[6] + '/{0:05d}'.format(idx) + ".png")

        ########################################################################
        reflect_beta = rendering_refl ** (1/gamma_corr) * rendering_beta
        combine_flash_img = render_RAW_image(rendering_trans_flash ** (1/gamma_corr) + reflect_beta)
        combine_noflash_img = render_RAW_image(rendering_trans_noflash ** (1/gamma_corr) + reflect_beta)
        combine_flash_img = torch.clamp(combine_flash_img, min=0., max=1.)
        combine_noflash_img = torch.clamp(combine_noflash_img, min=0., max=1.)
        # combine_flash_img = normalize(combine_flash_img)
        # combine_noflash_img = normalize(combine_noflash_img)
        # info(combine_flash_img, 'combine_flash_img')
        # info(combine_noflash_img, 'combine_noflash_img')

        torchvision.utils.save_image(combine_flash_img, render_folder[4] + '/{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(combine_noflash_img, render_folder[6] + '/{0:05d}'.format(idx) + ".png")

        ########################################################################

    
    # for each path in render_folder, read in all images and save as a gif
    import imageio
    for i, folder in enumerate(render_folder):
        images = []
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(".png"):
                # print(colored(os.path.join(folder, filename), 'red'))
                img = imageio.imread(os.path.join(folder, filename))
                # save it as jpeg and re-read it.
                cv2.imwrite(os.path.join(folder, filename).replace('.png', '.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                img = cv2.imread(os.path.join(folder, filename).replace('.png', '.jpg'))
                img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_LANCZOS4)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
        
        imageio.mimsave(os.path.join(model_path, 'video', f"{name_list[i]}.gif"), images, format='GIF', duration=40, loop=0)
        # exit()
        #     name_list = ['refl', 'beta', 'refl_beta', 'flash_trans', 'flash_comp', 'noflash_trans', 'noflash_comp']

        # if name_list[i] in ['refl', 'flash_trans', 'noflash_comp']:
        #     imageio.mimsave(f'{demo_gif_path}/{exp_name}_{name_list[i]}.gif', images, format='GIF', duration=40, loop=0)      
        #     print('Gif Saved at', f'{demo_gif_path}/{exp_name}_{name_list[i]}.gif')
        
        if name_list[i] in ['refl']:
            imageio.mimsave(f'{demo_gif_path}/{exp_name}_reflection.gif', images, format='GIF', duration=40, loop=0)      
            print('Gif Saved at', f'{demo_gif_path}/{exp_name}_reflection.gif')
        if name_list[i] in ['flash_trans']:
            imageio.mimsave(f'{demo_gif_path}/{exp_name}_transmission.gif', images, format='GIF', duration=40, loop=0)      
            print('Gif Saved at', f'{demo_gif_path}/{exp_name}_transmission.gif')
        if name_list[i] in ['noflash_comp']:
            imageio.mimsave(f'{demo_gif_path}/{exp_name}_composite_scene.gif', images, format='GIF', duration=40, loop=0)      
            print('Gif Saved at', f'{demo_gif_path}/{exp_name}_composite_scene.gif')



def render_sets(dataset : ModelParams, pipeline : PipelineParams, args):

    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.video:
            render_video(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTestCameras(),
                         gaussians, pipeline, background, args.fps)

        if not args.skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)
        if not args.skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)



'''
            render_set_ours(dataset.model_path, "train", scene_trans_noflash.loaded_iter, scene_trans_noflash.getTrainCameras()+scene_trans_noflash.getTestCameras(), gaussians_trans_noflash, gaussians_trans_flash, gaussians_reflect, gaussians_beta, pipe, background, args)
'''
# # For GS-version NeRFReN
# def render_set_ours(model_path, name, iteration, views, gaussians_trans_noflash, gaussians_trans_flash, gaussians_reflect, gaussians_beta, pipeline, background, args):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)
#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         image_trans = render(view, gaussians, pipeline, background)
#         image_reflect = render(view, gaussians_reflect, pipeline, background)
#         image_beta = render(view, gaussians_beta, pipeline, background)
#         image_combine = image_reflect["render"] * image_beta["render"] + image_trans ["render"]
#         gt = view.original_image[0:3, :, :]
#         torchvision.utils.save_image(image_trans["render"], os.path.join(render_path, view.image_name + '_trans.png'))
#         torchvision.utils.save_image(image_reflect["render"], os.path.join(render_path, view.image_name + '_reflect.png'))
#         torchvision.utils.save_image(image_combine, os.path.join(render_path, view.image_name + '_combine.png'))
#         torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--render_depth", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args)