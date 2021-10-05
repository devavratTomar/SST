import torch
import torch.nn as nn
import os

from utilities.util import load_network

from .encoder import Encoder

from .flow_model import FlowGenerator

from loss import GradientLoss
from pytorch_msssim import SSIM

class AutoEncoderFlow(nn.Module):
    def __init__(self, opt):
        super(AutoEncoderFlow, self).__init__()
        self.opt = opt

        # models
        self.E = Encoder(opt.nz)
        self.FG = FlowGenerator(opt.nz, opt.out_shape)

        if opt.continue_train:
            self.E = load_network(self.E, 'E', 'latest', opt.checkpoints_dir_pretrained)
            self.FG = load_network(self.FG, 'FlowGenerator', 'latest', opt.checkpoints_dir_pretrained)

        if len(opt.gpu_ids) > 0:
            self.E = self.E.cuda()
            self.FG = self.FG.cuda()
        
        # losses
        self.criterian_l1 = nn.L1Loss()
        self.criterian_ssim = SSIM(data_range=1.0, win_size=7, channel=1, spatial_dims=3, nonnegative_ssim=True)
        self.criterian_grad = GradientLoss()

    def create_optimizer(self):
        # try different learning rate for style mapper as in style gan paper 
        # maybe later.
        optimizer = torch.optim.Adam(list(self.E.parameters()) + list(self.FG.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))

        return optimizer

    def forward(self, base_img, base_label, real_img):

        autoencoder_loss = dict()

        # get encoder output for flow and appearance
        z = self.E(torch.cat([real_img, base_img], axis=1))

        # reshape z to match dim at d, w, h
        z = z[..., None, None, None]
        (predicted_img, predicted_label), field = self.FG(base_img, base_label, z)

        # l1 loss
        loss_l1 = self.opt.lambda_app*self.criterian_l1(predicted_img, real_img)
        autoencoder_loss.update({'AE_loss_l1': loss_l1})

        # appearance loss ssim
        ssim = -self.opt.lambda_ssim*torch.log(self.criterian_ssim((predicted_img + 1.0)/2, (real_img + 1.0)/2) + 1e-6)
        autoencoder_loss.update({'AE_loss_ssim': ssim})

        # gradient loss
        grad = self.opt.lambda_grad*self.criterian_grad(field)
        autoencoder_loss.update({'AE_loss_grad': grad})

        return autoencoder_loss, predicted_img, predicted_label

    def save(self, epoch):
        e_name = '%s_net_%s.pth' % (epoch, 'E')
        d_name = '%s_net_%s.pth' % (epoch, 'FlowGenerator')

        saved_model_path = os.path.join(self.opt.checkpoints_dir_pretrained, 'models')

        torch.save(self.E.state_dict(), os.path.join(saved_model_path, e_name))
        torch.save(self.FG.state_dict(), os.path.join(saved_model_path, d_name))
        
