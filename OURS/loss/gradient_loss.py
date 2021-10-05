import torch.nn as nn
import torch

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, volume):
        dy = torch.abs(volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :])
        dx = torch.abs(volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :])
        dz = torch.abs(volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1])

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        
        return d / 3.0

class GradientLossImage(nn.Module):
    def __init__(self):
        super(GradientLossImage, self).__init__()

    def get_gradient(self, img):
        img_dy = img[:, :, 1:, :, :] - img[:, :, :-1, :, :]
        img_dx = img[:, :, :, 1:, :] - img[:, :, :, :-1, :]
        img_dz = img[:, :, :, :, 1:] - img[:, :, :, :, :-1]

        img_grad = img_dy[:, :, :, 1:, :1]**2 + img_dx[:, :, 1:, :, 1:]**2 + img_dz[:, :, 1:, 1:, :]**2
        return img_grad
    
    def forward(self, input_img, target_img):
        input_img_grad = self.get_gradient(input_img)
        target_img_grad = self.get_gradient(target_img).detach()

        return torch.mean(torch.abs(input_img_grad - target_img_grad))