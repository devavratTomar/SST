import torch
import torch.nn as nn
import os

from utilities.util import load_network

from .generator import Generator
from .discriminator import Discriminator
from .encoder import Encoder
from .generator_style import StyleGenerator

from loss import GANLoss, GradientLoss, CrossCorrelationLoss
from pytorch_msssim import SSIM

class GANModel(nn.Module):
    def __init__(self, opt):
        super(GANModel, self).__init__()
        self.opt = opt

        # models
        self.E = Encoder(opt.nz)
        self.G = StyleGenerator(opt.flow_code_dim, opt.style_code_dim, opt.naf)
        self.D = Discriminator(opt.ndf)

        if opt.continue_train:
            self.E = load_network(self.E, 'E', 'latest', opt.checkpoints_dir)
            self.G = load_network(self.G, 'G', 'latest', opt.checkpoints_dir)
            self.D = load_network(self.D, 'D', 'latest', opt.checkpoints_dir)
        
        else:
            self.E = load_network(self.E, 'E', 'latest', os.path.join(opt.checkpoints_dir_pretrained))


        if len(opt.gpu_ids) > 0:
            self.E = self.E.cuda()
            self.G = self.G.cuda()
            self.D = self.D.cuda()
        
        # losses
        self.criterian_gan = GANLoss(opt.gan_type)
        self.criterian_gradient = GradientLoss()
        self.criterian_l1 = nn.L1Loss()
        self.criterian_ssim = SSIM(data_range=1.0, win_size=7, channel=1, spatial_dims=3, nonnegative_ssim=True)
        self.criterian_cc = CrossCorrelationLoss(win_size=7)

    def create_optimizers(self):
        # try different learning rate for style mapper as in style gan paper 
        # maybe later.
        optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))

        return optimizer_G, optimizer_D

    def discriminate(self, real_img, fake_img):
        fake_pred = self.D(fake_img)
        real_pred = self.D(real_img)

        return real_pred, fake_pred

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def forward(self, base_img, base_label, real_img, mode):

        if mode == 'generator':
            generator_loss = dict()

            # get encoder output for flow and appearance
            z_mu, z_log_var = self.E(real_img)
            z = self.reparameterize(z_mu, z_log_var)

            predicted_img, wrapped_img, wrapped_label, flow_field = self.G(base_img, base_label, z)

            # Gan loss
            gan_loss = self.criterian_gan(self.D(predicted_img), True)
            generator_loss.update({'G_loss': gan_loss})


            # gradient loss
            grad_loss = self.opt.lambda_grad*self.criterian_gradient(flow_field)
            generator_loss.update({'G_flow_reg_loss': grad_loss})


            # structure loss
            struct_loss_cc = self.opt.lambda_struct*self.criterian_cc(wrapped_img , real_img)
            generator_loss.update({'G_struct_loss': struct_loss_cc})

            struct_loss_ssim = -2*self.opt.lambda_ssim*torch.log(self.criterian_ssim((wrapped_img + 1.0)/2, (real_img + 1.0)/2) + 1e-6)
            generator_loss.update({'G_struct_loss_ssim': struct_loss_ssim})

            # appearance loss
            app_loss_l1 = self.opt.lambda_app*self.criterian_l1(predicted_img, real_img)
            generator_loss.update({'G_app_loss_l1': app_loss_l1})

            # appearance loss ssim
            app_loss_ssim = -self.opt.lambda_ssim*torch.log(self.criterian_ssim((predicted_img + 1.0)/2, (real_img + 1.0)/2) + 1e-6)
            generator_loss.update({'G_app_loss_ssim': app_loss_ssim})

            return generator_loss, predicted_img, wrapped_label

        else:
            discriminator_loss = dict()

            z_mu, z_log_var = self.E(real_img)
            z = self.reparameterize(z_mu, z_log_var)

            predicted_img, wrapped_img, wrapped_label, flow_field = self.G(base_img, base_label, z)

            # Gan loss
            real_pred, fake_pred = self.discriminate(real_img, predicted_img)

            loss = self.criterian_gan(real_pred, True) + self.criterian_gan(fake_pred, False)

            discriminator_loss.update({'D_loss': loss})

            return discriminator_loss

    def save(self, epoch):
        e_name = '%s_net_%s.pth' % (epoch, 'E')
        g_name = '%s_net_%s.pth' % (epoch, 'G')
        d_name = '%s_net_%s.pth' % (epoch, 'D')

        saved_model_path = os.path.join(self.opt.checkpoints_dir, 'models')

        torch.save(self.E.state_dict(), os.path.join(saved_model_path, e_name))
        torch.save(self.G.state_dict(), os.path.join(saved_model_path, g_name))
        torch.save(self.D.state_dict(), os.path.join(saved_model_path, d_name))
        
