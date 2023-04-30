from monai.networks.nets import ViTAutoEnc
import torch
import torch.nn as nn

class CustomViTAutoEnc(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, out_channels, pos_embed='conv', spatial_dims = 2):
        super(CustomViTAutoEnc, self).__init__()
        self.in_channels = in_channels # 10
        self.img_size = img_size # 128
        self.patch_size = patch_size #32
        self.out_channels = out_channels # 20
        self.pos_embed = pos_embed
        self.spatial_dims = spatial_dims
        self.vitautoenc = ViTAutoEnc(self.in_channels, self.img_size, self.patch_size, self.out_channels, self.pos_embed, self.spatial_dims)

    def forward(self, x):
        return self.vitautoenc(x)