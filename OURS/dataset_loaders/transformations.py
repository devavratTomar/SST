import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
import random

from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation

def create_gaussian_kernel(sigma, n_sigma_per_stride):
    t = np.linspace(-sigma*n_sigma_per_stride/2, sigma*n_sigma_per_stride/2, int(sigma * n_sigma_per_stride + 1))
    gauss_1_d = np.exp(-0.5*(t/sigma)**2)

    kernel = gauss_1_d[:, np.newaxis]*gauss_1_d[np.newaxis, :]
    kernel = kernel[np.newaxis, np.newaxis, ...]/np.sum(kernel)

    return torch.from_numpy(kernel).to(torch.float32)

class RandomColorJitter(object):
    def __init__(self, brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=[0.9, 1.1]):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, data):
        # data should be a list of 2 tensors of shape d x h x w
        q = data[None, ...].repeat(3, 1, 1, 1)
        
        brightness = random.uniform(self.brightness[0], self.brightness[1])
        contrast   = random.uniform(self.contrast[0], self.contrast[1])
        saturation = random.uniform(self.saturation[0], self.saturation[1])


        q = adjust_brightness(q, brightness)
        q = adjust_contrast(q, contrast)
        q = adjust_saturation(q, saturation)

        q = q[0] # only one channel
        return q

class RandomColorJitterPair(object):
    def __init__(self, brightness=[0.7, 1.3], contrast=[0.7, 1.3], saturation=[0.9, 1.1]):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, data):
        # data should be a list of 2 tensors of shape d x h x w
        q = data[0][None, ...].repeat(3, 1, 1, 1)
        k = data[1][None, ...].repeat(3, 1, 1, 1)

        brightness = random.uniform(self.brightness[0], self.brightness[1])
        contrast   = random.uniform(self.contrast[0], self.contrast[1])
        saturation = random.uniform(self.saturation[0], self.saturation[1])


        q = adjust_brightness(q, brightness)
        q = adjust_contrast(q, contrast)
        q = adjust_saturation(q, saturation)

        q = q[0] # only one channel

        k = adjust_brightness(k, brightness)
        k = adjust_contrast(k, contrast)
        k = adjust_saturation(k, saturation)

        k = k[0]

        return q, k
    

class RandomSmoothFlowPair3D(object):
    def __init__(self, size, flow_sigma, blur_sigma=5, n_sigma_per_stride=2):
        self.flow_sigma = flow_sigma
        self.blur_sigma = blur_sigma
        self.n_sigma_per_stride = n_sigma_per_stride
        self.padding = blur_sigma*n_sigma_per_stride//2

        self.size = size

        self.gauss_kernel = create_gaussian_kernel(blur_sigma, n_sigma_per_stride)
        self.gauss_kernel = self.gauss_kernel.repeat(size[0], size[0], 1, 1)

        vectors = [ torch.arange(0, s, dtype=torch.float32) for s in size ] 
        grids = torch.meshgrid(vectors) 
        grid  = torch.stack(grids)
        grid  = torch.unsqueeze(grid, 0)  #add batch
        self.grid = grid.type(torch.float32)

    @torch.no_grad()
    def __call__(self, data, sample_type='bilinear', p=0.5):
        # data input should be of shape [d x w x h]
        img = data.unsqueeze(0).unsqueeze(0)

        if random.random() >= (1.0 - p):
            flow = self.flow_sigma*torch.cat([nnf.conv2d(torch.randn([1] + self.size, dtype=torch.float32), self.gauss_kernel, padding=self.padding), 
                                            nnf.conv2d(torch.randn([1] + self.size, dtype=torch.float32), self.gauss_kernel, padding=self.padding),
                                            nnf.conv2d(torch.randn([1] + self.size, dtype=torch.float32), self.gauss_kernel, padding=self.padding)], dim=0)
            flow = flow.unsqueeze(0)
            new_locs = self.grid + flow

            # Need to normalize grid values to [-1, 1] for resampler
            for i in range(len(self.size)):
                new_locs[:,i,...] = 2*(new_locs[:,i,...]/(self.size[i]-1) - 0.5)

            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

            aug_img = nnf.grid_sample(img, new_locs, sample_type)

            # no batch dim needed
            aug_img = aug_img[0]
            return aug_img

        else:
            return img[0]




        

