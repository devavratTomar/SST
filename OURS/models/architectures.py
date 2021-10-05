import torch.nn as nn
import torch

from torch.distributions.normal import Normal


def image_to_noise_model(nz, in_ch=1):
    """Encoder architecture for input shape 128 x 160 x 160. nz is the dimension of the code required (Must be divisible by 16)
    """
    nf = nz//8

    return nn.Sequential(
        # input of shape 128 x 160 x 160
        nn.Conv3d(in_ch, nf, kernel_size=7, stride=4, padding=3),
        nn.InstanceNorm3d(nf),
        nn.LeakyReLU(0.2, True),

        # input of shape 32 x 40 x 40
        nn.Conv3d(nf, 2*nf, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm3d(2*nf),
        nn.LeakyReLU(0.2, True),

        # input of shape 16 x 20 x 20
        nn.Conv3d(2*nf, 4*nf, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm3d(4*nf),
        nn.LeakyReLU(0.2, True),

        # input of shape 8 x 10 x 10
        nn.Conv3d(4*nf, 8*nf, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm3d(8*nf),
        nn.LeakyReLU(0.2, True)
    )

def noise_to_image_model(nz):
    return nn.Sequential(
        # input of shape nz//2, 4, 5, 5
        nn.Upsample(scale_factor=2, mode='trilinear'),
        nn.Conv3d(nz//2, nz//4, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(nz//4),
        nn.LeakyReLU(0.2, True),

        # input of shape nz//4, 8, 10, 10
        nn.Upsample(scale_factor=2, mode='trilinear'),
        nn.Conv3d(nz//4, nz//8, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(nz//8),
        nn.LeakyReLU(0.2, True),

        nn.Upsample(scale_factor=2, mode='trilinear'),
        nn.Conv3d(nz//8, nz//16, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(nz//16),
        nn.LeakyReLU(0.2, True),

        nn.Upsample(scale_factor=2, mode='trilinear'),
        nn.Conv3d(nz//16, nz//32, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(nz//32),
        nn.LeakyReLU(0.2, True),

        nn.Upsample(scale_factor=2, mode='trilinear'),
        nn.Conv3d(nz//32, nz//64, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(nz//64),
        nn.LeakyReLU(0.2, True)
    )

class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDown, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.Conv3d(in_channels, out_channels, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ConvAdaInBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super(ConvAdaInBlock, self).__init__()
        self.n_features = out_channels
        
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
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

        x = x*gamma + beta
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
                                   nn.LeakyReLU(0.2, True),
                                   nn.Conv3d(dim, dim, 3, stride=1, padding=1),
                                   nn.LeakyReLU(0.2, True)
                                   )

    def forward(self, x):
        x_org = x
        residual = self.model(x)
        out = x_org + residual
        return out