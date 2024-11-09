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

import torch
from torch import nn
import numpy as np
import cv2
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from PIL import Image

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, image_noflash, gt_alpha_mask,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]),
                 scale=1.0, data_device = "cuda", depth_image = None, mask = None, bounds=None, image_path=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.depth_image = depth_image
        self.mask = mask
        self.bounds = bounds

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # BEGIN - CHM: added/modified to handle flash images
        self.original_image = image.to(self.data_device).cuda()
        self.original_image_noflash = image_noflash.to(self.data_device).cuda() if image_noflash is not None else None

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            self.original_image_noflash = self.original_image_noflash * gt_alpha_mask.to(self.data_device) if image_noflash is not None else None
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.original_image_noflash = self.original_image_noflash * torch.ones((1, self.image_height, self.image_width), device=self.data_device) if image_noflash is not None else None
        # END - CHM: added/modified to handle flash images

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def check_flash(self):
        # '_noflash' in self.image_name 
        # '_flash' in self.image_name
        return '_flash' in self.image_name

    def delete_image(self):
        del self.original_image
        del self.original_image_noflash
        del self.depth_image
        self.original_image = None
        self.original_image_noflash = None
        self.depth_image = None


class PseudoCamera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, width, height, trans=np.array([0.0, 0.0, 0.0]), scale=1.0 ):
        super(PseudoCamera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]



class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

