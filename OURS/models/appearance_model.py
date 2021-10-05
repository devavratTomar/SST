import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .architectures import ConvBlockDown, ResBlock, ConvAdaInBlock 

from torch.distributions.normal import Normal

class ApperanceModelStyle(nn.Module):
    def __init__(self, nf, style_dim):
        super(ApperanceModelStyle, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2, True)
        )

        self.init_layer = nn.Conv3d(1, nf, 7, stride=1, padding=3)
        down_layers = [nn.LeakyReLU(0.2, True)]

        prev_f = nf
        for i in range(2):
            ch = 2*nf
            down_layers += [ConvBlockDown(prev_f, ch)]
            prev_f = ch
        
        # resnet block
        res_blocks = []
        for i in range(2):
            res_blocks += [ResBlock(prev_f)]

        self.content_encoder = nn.Sequential(*(down_layers + res_blocks))

        # style decoder
        style_decoder = []
        for i in range(2):
            ch = 2*nf
            style_decoder += [ConvAdaInBlock(prev_f, ch, style_dim)]
            prev_f = ch
        
        style_decoder += [ConvAdaInBlock(prev_f, nf, style_dim)]
        
        self.style_decoder = nn.ModuleList(style_decoder)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear')
        
        # final layer
        self.final_layer = nn.Conv3d(nf, 1, 3, stride=1, padding=1)

    def forward(self, x, style):
        style = self.mlp(style)

        x = self.init_layer(x)
        residual = x

        x = self.content_encoder(x)
        
        for i in range(len(self.style_decoder)-1):
            x = self.style_decoder[i](x, style)
            x = self.up(x)
        
        x = self.style_decoder[-1](x, style)
        # x = x + residual
        
        x = self.final_layer(x)

        out = torch.tanh(x)

        return out





