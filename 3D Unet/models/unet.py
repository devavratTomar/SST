import torch
import torch.nn as nn

def getNormLayer(och, norm="batch"):
    if norm == "batch":
        norm_layer = nn.BatchNorm2d(och, affine=False)
    elif norm == "inst":
        norm_layer = nn.InstanceNorm2d(och, track_running_stats=False)
    else:
        norm_layer = None
    return norm_layer


class Double_conv(nn.Module):
    """
    Apply a double convolution on the input x
    args:
        - ich: Number of channel of the input
        - och: Number of channel of the output
    """

    def __init__(self, ich, och, norm="batch"):
        super(Double_conv, self).__init__()

        norm_layer = getNormLayer(och, norm)
        """
        self.conv = nn.Sequential(nn.Conv2d(ich, och, 3, 1, 1, bias=False),
                                  norm_layer,
                                  nn.LeakyReLU(0.1),
                                  nn.Conv2d(och, och, 3, 1, 1, bias=False),
                                  norm_layer,
                                  nn.LeakyReLU(0.1))
        """
        self.conv = nn.Sequential(nn.Conv2d(ich, och, 3, 1, 1, bias=False),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(och, och, 3, 1, 1, bias=False),
                                  nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    """
    Apply an encoding convolution, divides the image size by two
    args:
        - ich: Number of channel of the input
        - och: Number of channel of the output
    """

    def __init__(self, ich, och, downsample=False, norm="batch"):
        super(Down, self).__init__()

        if downsample:
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = nn.Conv2d(ich, ich, 4, 2, 1, bias=False)
        self.conv = Double_conv(ich, och, norm=norm)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Apply a decoding deconvolution, multiplies the image size by two
    args:
        - ich: Number of channel of the input
        - och: Number of channel of the output
    """

    def __init__(self, ich, och, norm="batch"):
        super(Up, self).__init__()

        norm_layer = getNormLayer(ich // 2, norm)
        """
        self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                nn.Conv2d(ich, ich // 2, kernel_size=1, bias=False),
                                norm_layer)
        """
        self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True),
                                nn.Conv2d(ich, ich // 2, kernel_size=1, bias=False))
                                
        self.conv = Double_conv(och + ich // 2, och, norm=norm)

    def forward(self, enc_layer, dec_layer):
        dec = self.up(dec_layer)
        x = torch.cat([enc_layer, dec], dim=1)
        x = self.conv(x)
        return x


class UNET(nn.Module):
    """
    Implements a unet network with convolutional encoding and deconvolutional decoding.
    The forward method implements skip connections between encoder and decoder to keep consistency in the features
    arg:
        n_classes: The number of channel output
    """

    def __init__(self, ich, n_classes, downsample=False, norm="batch"):
        super(UNET, self).__init__()

        self.d1 = Double_conv(ich, 32, norm=norm)
        self.d2 = Down(32, 64, downsample=downsample, norm=norm)
        self.d3 = Down(64, 128, downsample=downsample, norm=norm)
        self.d4 = Down(128, 256, downsample=downsample, norm=norm)
        self.d5 = Down(256, 512, downsample=downsample, norm=norm)

        self.u2 = Up(512, 256)
        self.u3 = Up(256, 128)
        self.u4 = Up(128, 64)
        self.u5 = Up(64, 32)

        self.final = nn.Conv2d(32, n_classes, 1, 1, 0, bias=False)

    def forward(self, x):
        e1 = self.d1(x)
        e2 = self.d2(e1)
        e3 = self.d3(e2)
        e4 = self.d4(e3)
        x = self.d5(e4)

        x = self.u2(e4, x)
        x = self.u3(e3, x)
        x = self.u4(e2, x)
        x = self.u5(e1, x)
        x = self.final(x)

        return x