import torch.nn as nn
import torch

from torch.distributions.normal import Normal


def image_to_noise_model(nz=1024):
    """Encoder architecture for input shape 128 x 160 x 160. nz is the dimension of the code required (Must be divisible by 16)
    """
    nf = nz//16

    return nn.Sequential(
        # input of shape 64 x 80 x 80
        nn.Conv3d(2, nf, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm3d(nf),
        nn.ReLU(True),

        # input of shape 32 x 40 x 40
        nn.Conv3d(nf, 2*nf, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm3d(2*nf),
        nn.ReLU(True),

        # input of shape 16 x 20 x 20
        nn.Conv3d(2*nf, 4*nf, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm3d(4*nf),
        nn.ReLU(True),

        # input of shape 8 x 10 x 10
        nn.Conv3d(4*nf, 8*nf, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm3d(8*nf),
        nn.ReLU(True),

        # # input of shape 4 x 5 x 5
        # nn.Conv3d(8*nf, 16*nf, kernel_size=3, stride=2, padding=1),
        # nn.InstanceNorm3d(16*nf),
        # nn.ReLU(True)

        # output of shape 4 x 5 x 5
        # Adaptive avg pool
        # nn.AdaptiveAvgPool3d(1),

        # output of size (nz, 1, 1, 1)
        # # Linear projection layer
        # nn.Linear(nz, nz, bias=False)
    )

def noise_to_image_model(nz=1024):
    return nn.Sequential(
        # input shape of (nz, 1, 1, 1)
        nn.ConvTranspose3d(nz, nz//2, kernel_size=[4, 5, 5], stride=[1, 1, 1], padding=[0, 0, 0]),
        nn.InstanceNorm3d(nz//2),
        nn.ReLU(True),

        # input size of (512, 4, 5, 5)
        nn.ConvTranspose3d(nz//2, nz//4, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=[1, 1, 1]),
        nn.InstanceNorm3d(nz//4),
        nn.ReLU(True),

        # input size of (256, 8, 10, 10)
        nn.ConvTranspose3d(nz//4, nz//8, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=[1, 1, 1]),
        nn.InstanceNorm3d(nz//8),
        nn.ReLU(True),

        # input size of (128, 16, 20, 20)
        nn.ConvTranspose3d(nz//8, nz//16, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=[1, 1, 1]),
        nn.InstanceNorm3d(nz//16),
        nn.ReLU(True),

        # input size of (64, 32, 40, 40)
        nn.ConvTranspose3d(nz//16, nz//32, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=[1, 1, 1]),
        nn.InstanceNorm3d(nz//32),
        nn.ReLU(True),

        # input size of (32, 64, 80, 80)
        # nn.ConvTranspose3d(nz//32, nz//64, kernel_size=[4, 4, 4], stride=[2, 2, 2], padding=[1, 1, 1]),
        # nn.InstanceNorm3d(nz//64),
        # nn.ReLU(True)
        # output shape is (16, 128, 160, 160)
        # nn.Conv3d(nz//64, 1, 1, 1, 0),
        # nn.Tanh()
        
        # final output shape is 1, 128, 160, 160
    )

class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDown, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class ConvAdaInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super(ConvAdaInBlock, self).__init__()
        self.n_features = out_channels
        
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        nd = Normal(0, 1e-4)

        self.gamma_layer = nn.Linear(style_dim, out_channels)
        self.gamma_layer.weight = nn.Parameter(nd.sample(self.gamma_layer.weight.shape))
        self.gamma_layer.bias   = nn.Parameter(torch.zeros(self.gamma_layer.bias.shape))
        
        self.beta_layer  = nn.Linear(style_dim, out_channels)
        self.beta_layer.weight = nn.Parameter(nd.sample(self.beta_layer.weight.shape))
        self.beta_layer.bias   = nn.Parameter(torch.zeros(self.beta_layer.bias.shape))

    def adain(self, x, style):
        # first normalize x
        mean, var = torch.mean(x, dim=[2, 3, 4], keepdim=True), torch.var(x, dim=[2, 3, 4], keepdim=True)
        x = (x - mean)/(torch.sqrt(var + 1e-5))

        # apply style
        gamma = 1.0 + self.gamma_layer(style)
        beta  = self.beta_layer(style)

        # reshape them to match the features
        gamma = gamma.view(-1, self.n_features, 1, 1, 1)
        beta  = beta.view(-1, self.n_features, 1, 1, 1)

        x = x*(1.0 +gamma) + beta
        return x

    def forward(self, x, style):
        # perform convolution
        x = self.conv(x)

        # adain
        x = self.adain(x, style)

        # activation
        x = self.act(x)

        return x
        

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.model = nn.Sequential(nn.Conv3d(dim, dim, 3, stride=1, padding=1),
                                   nn.ReLU(True),
                                   nn.Conv3d(dim, dim, 3, stride=1, padding=1),
                                   nn.ReLU(True)
                                   )

    def forward(self, x):
        x_org = x
        residual = self.model(x)
        out = x_org + residual
        return out