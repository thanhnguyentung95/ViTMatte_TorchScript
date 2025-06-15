from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os

from detectron2.structures import ImageList

class ViTMatte(nn.Module):
    def __init__(self,
                 *,
                 backbone,
                 criterion,
                 pixel_mean,
                 pixel_std,
                 input_format,
                 size_divisibility,
                 decoder,
                 ):
        super(ViTMatte, self).__init__()
        self.backbone = backbone
        self.criterion = criterion
        self.input_format = input_format
        self.size_divisibility = size_divisibility
        self.decoder = decoder
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
    
    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: Dict[str, torch.Tensor]):
        images, targets, H, W = self.preprocess_inputs(batched_inputs)

        features = self.backbone(images)
        outputs = self.decoder(features, images)  

        outputs = outputs[:,:,:H,:W]
        return outputs



    def preprocess_inputs(self, batched_inputs: Dict[str, torch.Tensor]):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs['image'].to(self.device)
        trimap = batched_inputs['trimap'].to(self.device)
        images = (images - self.pixel_mean) / self.pixel_std

        images = torch.cat((images, trimap), dim=1)
        
        B, C, H, W = images.shape
        if H % 32 != 0 or W % 32 != 0:
            new_H = (32 - H % 32) + H
            new_W = (32 - W % 32) + W
            new_images = torch.zeros((B, C, new_H, new_W), device=self.device)
            new_images[:, :, :H, :W] = images
            images = new_images
        # No need for `else`, images is valid in both cases now


        phas = None

        return images, dict(phas=phas), H, W