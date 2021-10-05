import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .architectures import ConvBlockDown, ResBlock, ConvAdaInBlock 

from torch.distributions.normal import Normal

class ApperanceModelStyle(nn.Module):
    def __init__(self, nf, style_dim):
        super(ApperanceModelStyle, self).__init__()

        self.init_layer = nn.Conv3d(1, nf, 3, stride=1, padding=1)

        down_layers = []
        for i in range(2):
            ch = (2**i)*nf
            down_layers += [ConvBlockDown(ch, 2*ch)]
        
        # resnet block
        res_blocks = []
        for i in range(3):
            res_blocks += [ResBlock(2*ch)]

        self.content_encoder = nn.Sequential(*(down_layers + res_blocks))

        # style decoder
        style_decoder = []
        for i in range(2):
            ch = (2**(2-i))*nf
            style_decoder += [ConvAdaInBlock(ch, ch//2),
                              nn.Upsample(scale_factor=2, mode='trilinear')]
        
        style_decoder += [ConvAdaInBlock(nf, nf//2)]
        
        self.style_decoder = nn.Sequential(*style_decoder)
        
        # final layer
        self.final_layer = nn.Conv3d(nf//2, 1, 3, stride=1, padding=1)

    def forward(self, x, style):
        x_org = x
        x = self.init_layer(x)
        x = self.content_encoder(x)

        x = self.style_decoder(x, style)
        x = self.final_layer(x)

        out = x_org + 0.1*x
        return out





