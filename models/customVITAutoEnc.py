from monai.networks.nets import ViTAutoEnc


class CustomViTAutoEnc():
    def __init__(self, in_channels, img_size, patch_size, out_channels):
        super(CustomViTAutoEnc, self).__init__()
        self.in_channels = in_channels # 10
        self.img_size = img_size # 128
        self.patch_size = patch_size #32
        self.out_channels = out_channels # 20
        self.vitautoenc = ViTAutoEnc(self.in_channels, self.img_size, self.patch_size, self.out_channels)

    def forward(self, x):
        return self.vitautoenc(x)