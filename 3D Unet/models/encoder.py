import torch
import torch.nn as nn
from math import ceil

from .architectures import image_to_noise_model, noise_to_image_model

class Encoder(nn.Module):
    def __init__(self, nz):
        super(Encoder, self).__init__()
        self.backbone = image_to_noise_model(nz)
        self.fc = nn.Sequential(nn.Linear(50*nz, nz), nn.ReLU(True), nn.Linear(nz, nz))

    def forward(self, img):
        x = self.backbone(img)
        # (batch size, nz)
        x = x.view(img.shape[0], -1)
        
        x = self.fc(x)
        return x

class EncoderStyle(nn.Module):
    def __init__(self, style_dim):
        super(EncoderStyle, self).__init__()
        self.backbone = image_to_noise_model(1024)

        # output of shape 512 x 4 x 5 x 5
        self.conv1x1 = nn.Conv3d(512, 256, kernel_size=1, stride=1, bias=False)

        # adaptive pooling across spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool3d(1)

        # final fc
        self.fc = nn.Sequential(nn.Linear(256, style_dim),
                                nn.LeakyReLU(0.2, True),
                                nn.Linear(style_dim, style_dim, bias=False))

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1x1(x)
        x = self.adaptive_pool(x)

        # output of shape batch_size x 512 x 1 x 1 x 1
        x = x.view(x.shape[0], -1)

        x = self.fc(x)

        return x


