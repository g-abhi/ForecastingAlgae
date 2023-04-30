from monai.networks.nets import ViTAutoEnc
import torch
import torch.nn as nn

class CustomViTAutoEnc(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, out_channels=1, deconv_chns=16, hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12, pos_embed='conv', dropout_rate=0.0, spatial_dims=2):
        super(CustomViTAutoEnc, self).__init__()
        self.in_channels = in_channels # 10
        self.img_size = img_size #(128, 128)
        self.patch_size = patch_size # (16, 16)
        self.out_channels = out_channels # 20
        self.deconv_chns = deconv_chns
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pos_embed = pos_embed
        self.dropout_rate = dropout_rate
        self.spatial_dims = spatial_dims
        self.vitautoenc = ViTAutoEnc(
            in_channels = self.in_channels,
            img_size = self.img_size,
            patch_size = self.patch_size,
            out_channels = self.out_channels,
            deconv_chns = self.deconv_chns,
            hidden_size = self.hidden_size,
            mlp_dim = self.mlp_dim,
            num_layers = self.num_layers,
            num_heads = self.num_heads,
            pos_embed = self.pos_embed,
            dropout_rate = self.dropout_rate,
            spatial_dims = self.spatial_dims
        )

    def forward(self, x):
        return self.vitautoenc(x)[0] # this only returns X_out for loss calculation

    def forward_hiddenstates(self, x):
        '''For model inference.'''
        return self.vitautoenc(x) # return both X_out and hidden states