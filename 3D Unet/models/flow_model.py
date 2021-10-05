import torch
import torch.nn as nn
import torch.nn.functional as nnf
from .architectures import noise_to_image_model
from torch.distributions.normal import Normal

class FlowGenerator(nn.Module):
    def __init__(self, nz, out_shape):
        """
        This module generates the 2D or 3D flow from a random vector. The architecture is inspired by DCGAN.
        Assume the output shape is of shape d x h x w = 128 x 256 x 256

        :param dim: input dimension 2D or 3D.
        :param nz:  dimension of noise vector
        :param ngf: feature map dimension
        """
        super(FlowGenerator, self).__init__()

        conv_fn = getattr(nn, 'Conv%dd' % 3)

        self.main = noise_to_image_model(nz)
        
        # input of shape nz//64, 128, 160, 160
        self.flow = conv_fn(nz//32, 3, kernel_size=3, padding=1)

#        Make flow weights + bias small.
        # nd = Normal(0, 1e-3)
        # self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.spatial_transformer = SpatialTransformer(out_shape)

        # learnable scale_weight
        self.scale_weight = nn.Parameter(torch.tensor(1e-1, requires_grad=True))

    def forward(self, img, label, noise):
        x = self.main(noise)
        x = torch.abs(self.scale_weight)*self.flow(x)
        tgt = self.spatial_transformer(img, label, x)
        print(self.scale_weight)
        return tgt, x


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, size):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [ torch.arange(0, s) for s in size ] 
        grids = torch.meshgrid(vectors) 
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        grid = grid.type(torch.float32)
        self.register_buffer('grid', grid)

    def forward(self, src_img, src_label, flow):   
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow 

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) 
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1) 
            new_locs = new_locs[..., [2,1,0]]
        
        return nnf.grid_sample(src_img, new_locs, 'bilinear', 'border'), nnf.grid_sample(src_label, new_locs, 'nearest', 'border')