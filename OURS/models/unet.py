import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, 5, stride, 2)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class Unet(nn.Module):
    def __init__(self, encoder=[16, 32, 32, 32], decoder=[32, 32, 32, 16, 16]):
        super(Unet, self).__init__()
        self.enc_nf, self.dec_nf = encoder, decoder

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder
        prev_nf = 2 # initial number of channels
        self.downarm = nn.ModuleList()

        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(prev_nf, nf, stride=2))
            prev_nf = nf

        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()

        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(prev_nf, nf, stride=1))
            prev_nf = nf
        
        # final convolution
        self.final = nn.Conv3d(prev_nf, 3, kernel_size=5, padding=2)
        
        nd = Normal(0, 1e-3)
        self.final.weight = nn.Parameter(nd.sample(self.final.weight.shape))
        self.final.bias = nn.Parameter(torch.zeros(self.final.bias.shape))

    def forward(self, x):
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

         # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return self.final(x)