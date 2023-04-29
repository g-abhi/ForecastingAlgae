import torch
import torch.nn as nn

class Generator(torch.nn.Module):
    def __init__(self, image_size, latent_z_size,out_channels) :
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(latent_z_size,out_channels,kernel_size=4,stride=1, padding=0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels,out_channels//2,kernel_size=4,stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels//2,out_channels//4,kernel_size=4,stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(out_channels//4,image_size,kernel_size=4,stride=2, padding=1, bias=False),
            nn.Tanh(),

        )
    def forward(self, input):
        output = self.gen(input)
        return output

class Discriminator(torch.nn.Module):
    def __init__(self, image_size, latent_z_size,out_channels_d) :
        self.dis = nn.Sequential(
            nn.Conv2d(image_size,out_channels_d,kernel_size=4,stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(out_channels_d,out_channels_d * 2,kernel_size=4,stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(out_channels_d * 2,out_channels_d * 4,kernel_size=4,stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(out_channels_d * 4, 1, 4, 1, 0, bias=False)

        )
    def forward(self, input):
        output = self.dis(input)
        output = output.mean(0).view(1)
        return output
    