### adeverserial autoencoder for generating flow from noise
import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class ConvTransposeBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True):
        super().__init__()

        self.main = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class FlowDiscriminator(nn.Module):
    """
    PatchGan based discriminator
    """
    def __init__(self, features=[32, 64, 128, 256]):
        super(FlowDiscriminator, self).__init__()

        layers = []
        prev_nf = 3
        for nf in features:
            layers += [ConvBlock(prev_nf, nf, kernel_size=4, stride=2, padding=1)]
            prev_nf = nf

        layers += [nn.Conv3d(prev_nf, 1, kernel_size=4, stride=1, padding=1, bias=False)]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class FlowEncoder(nn.Module):
    def __init__(self, features=[32, 64, 128, 256]):
        super(FlowEncoder, self).__init__()

        conv_layers = []
        prev_nf = 3

        for nf in features:
            conv_layers += [ConvBlock(prev_nf, nf, kernel_size=5, stride=2, padding=2)]
            prev_nf = nf
        
        conv_layers += [nn.Conv3d(prev_nf, 64, kernel_size=4, stride=2, padding=1, bias=False)]

        self.conv_model = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv_model(x)
        return x


class FlowGenerator(nn.Module):
    def __init__(self):
        super(FlowGenerator, self).__init__()
        features=[512//2, 512//4, 512//8, 512//16, 512//16]
        conv_model = []
        prev_nf = 64
        for i, nf in enumerate(features):
            conv_model += [ConvTransposeBlock(prev_nf, nf, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=[1, 1, 1])]
            prev_nf = nf

        conv_model += [nn.Conv3d(prev_nf, 3, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1])]

        self.conv_model = nn.Sequential(*conv_model)

    def forward(self, code):
        x = code
        x = self.conv_model(x)
        return x