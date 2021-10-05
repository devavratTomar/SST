import torch.nn as nn

from .architectures import image_to_noise_model

class Encoder(nn.Module):
    def __init__(self, nz):
        super(Encoder, self).__init__()
        self.backbone = image_to_noise_model(nz, 2)
        self.conv1x1 = nn.Conv3d(nz, nz//2, kernel_size=1, stride=1, bias=False)
    def forward(self, img):
        x = self.backbone(img)
        x = self.conv1x1(x)
        return x

class EncoderStyle(nn.Module):
    def __init__(self, style_dim):
        super(EncoderStyle, self).__init__()
        self.backbone = image_to_noise_model(128, 1)

        # input of shape 128 x 4 x 5 x 5
        self.conv1x1 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.fc = nn.Sequential(nn.Linear(64*4*5*5, style_dim),
                                nn.LeakyReLU(0.2, True),
                                nn.Linear(style_dim, style_dim),
                                nn.LeakyReLU(0.2, True),
                                nn.Linear(style_dim, style_dim, bias=False))

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1x1(x)

        x = x.view(x.shape[0], -1)

        x = self.fc(x)

        # normalize the style embedding to have unit norm
        x = nn.functional.normalize(x, dim=1)
        return x
