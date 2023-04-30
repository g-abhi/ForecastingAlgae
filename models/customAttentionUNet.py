from monai.networks.nets import AttentionUNet
import torch
import torch.nn as nn

class CustomAttentionUNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides, kernel_size=3, up_kernel_size=3, dropout=0.0):
        super(CustomAttentionUNet, self).__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.dropout = dropout
        self.attn_unet = AttentionUNet(
            spatial_dims = self.spatial_dims,
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            channels = self.channels,
            strides = self.strides,
            kernel_size = self.kernel_size,
            up_kernel_size = self.up_kernel_size,
            dropout = self.dropout
        )

    def forward(self, x):
        return self.attn_unet(x)