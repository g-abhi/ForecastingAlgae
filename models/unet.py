'''
3D time-dependent DL model (Conv3D backbone, UNET baseline) for forward-forecasting.
Expects 5D input of the form: torch.Size([1, timestep_in, Channels (1), X, Y]) 
'''

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.Sequential(
            conv_block(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.middle = conv_block(64, 128)

        self.upconv = upconv_block(128, 64)

        self.decoder = conv_block(64, out_channels)

    def forward(self, x):
        enc = self.encoder(x)
        middle = self.middle(enc)
        upconv = self.upconv(middle)
        dec = self.decoder(upconv)
        
        # Add padding if necessary
        target_h, target_w = x.size(2), x.size(3)
        pad_h = target_h - dec.size(2)
        pad_w = target_w - dec.size(3)

        if pad_h > 0 or pad_w > 0:
            dec = nn.functional.pad(dec, (0, pad_w, 0, pad_h))

        return dec