from monai.networks.nets import UNet
import torch
import torch.nn as nn

class CustomUNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides, kernel_size=3, up_kernel_size=3, num_res_units=0, act='PRELU', norm='INSTANCE', dropout=0.0, bias=True, adn_ordering='NDA', dimensions=None):
        super(CustomUNet, self).__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides 
        self.kernel_size = kernel_size 
        self.up_kernel_size = up_kernel_size 
        self.num_res_units = num_res_units 
        self.act = act
        self.norm = norm 
        self.dropout = dropout 
        self.bias = bias 
        self.adn_ordering = adn_ordering 
        self.dimensions = dimensions
        self.unet = UNet(
            spatial_dims = self.spatial_dims,
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            channels = self.channels,
            strides = self.strides,
            kernel_size = self.kernel_size,
            up_kernel_size = self.up_kernel_size,
            num_res_units = self.num_res_units,
            act = self.act,
            norm = self.norm,
            dropout = self.dropout,
            bias = self.bias,
            adn_ordering = self.adn_ordering,
            dimensions = self.dimensions
        )

    def forward(self, x):
        return self.unet(x)