import torch
import torch.nn as nn
import os

from models import FlowGenerator, FlowDiscriminator, DisLatentCode, SpatialTransformer, FlowEncoder
from utilities.util import load_network

from dataset_loaders import FlowSampler

from loss import GANLoss, GradientLoss
import torch.nn.functional as nnf

from pytorch_msssim import SSIM


class TrainerFlowAAE(object):
    def __init__(self, opt):
        self.opt = opt

        self.flow_encoder = FlowEncoder().cuda()
        self.flow_generator = FlowGenerator().cuda()
        self.latent_discriminator = DisLatentCode(4*5*5*64, n_layers=2, h_dim=256).cuda()
        self.spatial_transformer = SpatialTransformer(opt.out_shape).cuda()

        if opt.continue_train:
            self.flow_encoder = load_network(self.flow_encoder, 'FlowEncoder', 'latest', opt.checkpoints_dir_pretrained)
            self.flow_generator = load_network(self.flow_generator, 'FlowGenerator', 'latest', opt.checkpoints_dir_pretrained)
            self.latent_discriminator = load_network(self.latent_discriminator, 'FlowLatentDiscriminator', 'latest', opt.checkpoints_dir_pretrained)
        
        if len(opt.gpu_ids) > 0:
            self.flow_generator = nn.DataParallel(self.flow_generator, device_ids = opt.gpu_ids)
            self.flow_encoder   = nn.DataParallel(self.flow_encoder, device_ids = opt.gpu_ids)
            self.latent_discriminator = nn.DataParallel(self.latent_discriminator, device_ids = opt.gpu_ids)
            self.spatial_transformer = nn.DataParallel(self.spatial_transformer, device_ids=opt.gpu_ids)
        
        ae_params = list(self.flow_encoder.parameters()) + list(self.flow_generator.parameters())
        latent_params = list(self.latent_discriminator.parameters())

        self.optimizer_ae = torch.optim.Adam(ae_params, lr=self.opt.lr, betas=(0.9, 0.999), weight_decay=self.opt.weight_decay)
        self.optimzer_latent = torch.optim.Adam(latent_params, lr=self.opt.lr, betas=(0.5, 0.999), weight_decay=self.opt.weight_decay)

        self.criterian_gan = GANLoss('lsgan')
        self.ssim_loss     = SSIM(1.0, win_size=7, channel=1, spatial_dims=3, nonnegative_ssim=True)
        self.l1_loss       = nn.L1Loss()

    def run_step(self, base_img, base_seg, real_flow_feild):
        # train ae
        self.optimizer_ae.zero_grad()
        flowcode = self.flow_encoder(real_flow_feild)

        fake_flow_feild = self.flow_generator(flowcode)

        loss_l1 = self.l1_loss(fake_flow_feild, real_flow_feild)
        
        fake_img, fake_label = self.spatial_transformer(base_img, base_seg, fake_flow_feild)
        real_img, real_label = self.spatial_transformer(base_img, base_seg, real_flow_feild)

        loss_ssim = 1.0 - self.ssim_loss((fake_img + 1.0)/2.0, (real_img + 1.0)/2.0)

        loss_adver = 0.05*self.criterian_gan(self.latent_discriminator(flowcode), True)

        loss_ae = loss_ssim + loss_l1 + loss_adver
        loss_ae.backward()
        self.optimizer_ae.step()

        # train discriminator
        self.optimzer_latent.zero_grad()
        noise = torch.randn_like(flowcode).cuda()
        loss_latent = 0.05*self.criterian_gan(self.latent_discriminator(flowcode.detach()), False) + 0.05*self.criterian_gan(self.latent_discriminator(noise), True)
        loss_latent.backward()
        self.optimzer_latent.step()

        ## evaluations..
        self.all_losses = {
            'loss_flow_recon':loss_l1.detach(),
            'loss_img_recon':loss_ssim.detach(),
            'loss_latent_g':loss_adver.detach(),
            'loss_latent_d': loss_latent.detach()
        }


        self.real_img, self.real_label = real_img.detach(), real_label.detach()
        self.fake_img, self.fake_label = fake_img.detach(), fake_label.detach()

    def get_latest_losses(self):
        return self.all_losses

    def update_learning_rate(self, lr):
        for param_group in self.optimizer_ae.param_groups:
            param_group['lr'] = lr
        
        for param_group in self.optimzer_latent.param_groups:
            param_group['lr'] = lr

    def save(self, epoch):
        encoder_name   = '%s_net_%s.pth' % (epoch, 'FlowEncoder')
        generator_name = '%s_net_%s.pth' % (epoch, 'FlowGenerator')
        latent_name    = '%s_net_%s.pth' %(epoch, 'FlowLatentDiscriminator')

        saved_model_path = os.path.join(self.opt.checkpoints_dir_pretrained, 'models')
        
        torch.save(self.flow_encoder.module.state_dict(), os.path.join(saved_model_path, encoder_name))
        torch.save(self.flow_generator.module.state_dict(), os.path.join(saved_model_path, generator_name))
        torch.save(self.latent_discriminator.module.state_dict(), os.path.join(saved_model_path, latent_name))



class TrainerFlowGAN(object):
    def __init__(self, opt):
        self.opt = opt

        self.generator        = FlowGenerator().cuda()
        self.discriminator    = FlowDiscriminator().cuda()

        self.spatial_transformer = SpatialTransformer(opt.out_shape).cuda()

        if opt.continue_train:
            self.generator     = load_network(self.generator, 'FlowGenerator', 'latest', opt.checkpoints_dir_pretrained)
            self.discriminator = load_network(self.discriminator, 'FlowDiscriminator', 'latest', opt.checkpoints_dir_pretrained)

        if len(opt.gpu_ids) > 0:
            self.generator           = nn.DataParallel(self.generator, device_ids=opt.gpu_ids)
            self.discriminator       = nn.DataParallel(self.discriminator, device_ids=opt.gpu_ids)
            self.spatial_transformer = nn.DataParallel(self.spatial_transformer, device_ids=opt.gpu_ids)
        
        g_params = list(self.generator.parameters())
        d_params = list(self.discriminator.parameters())
        
        self.optimizer_g = torch.optim.Adam(g_params, lr=self.opt.lr, betas=(0.5, 0.999), weight_decay=self.opt.weight_decay)
        self.optimizer_d = torch.optim.Adam(d_params, lr=self.opt.lr, betas=(0.5, 0.999), weight_decay=self.opt.weight_decay)

        self.criterian_gan = GANLoss('lsgan')
        self.grad_loss     = GradientLoss()

    def run_step(self, base_img, base_seg, real_flow_feild):
        # train generator
        # flow_code
        flow_code = torch.randn(real_flow_feild.shape[0], 64, 4, 5, 5).cuda()

        self.optimizer_g.zero_grad()

        fake_flow_field = self.generator(flow_code)
        fake_predict    = self.discriminator(fake_flow_field)
        
        # averserail loss
        loss_gen = self.criterian_gan(fake_predict, True)
        loss_grad = self.grad_loss(fake_flow_field)
        loss_all = loss_gen + 0.05*loss_grad
        loss_all.backward()
        self.optimizer_g.step()

        # train discriminator
        self.optimizer_d.zero_grad()
        real_predict = self.discriminator(real_flow_feild)
        fake_predict = self.discriminator(self.generator(flow_code))

        # adverserail loss
        loss_dis = self.criterian_gan(real_predict, True) + self.criterian_gan(fake_predict, False)
        loss_dis.backward()
        self.optimizer_d.step()

        ## evaluations..
        self.all_losses = {
            'loss_gen':loss_gen.detach(),
            'loss_dis':loss_dis.detach(),
            'loss_grad':0.05*loss_grad.detach()
        }


        self.real_img, self.real_label = self.spatial_transformer(base_img, base_seg, real_flow_feild)
        self.fake_img, self.fake_label = self.spatial_transformer(base_img, base_seg, fake_flow_field.detach())
        
    
    def get_latest_losses(self):
        return self.all_losses

    def update_learning_rate(self, lr):
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = lr
        
        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] = lr

    def save(self, epoch):
        g_name   = '%s_net_%s.pth' % (epoch, 'FlowGenerator')
        d_name   = '%s_net_%s.pth' % (epoch, 'FlowDiscriminator')

        saved_model_path = os.path.join(self.opt.checkpoints_dir_pretrained, 'models')
        torch.save(self.generator.module.state_dict(), os.path.join(saved_model_path, g_name))
        torch.save(self.discriminator.module.state_dict(), os.path.join(saved_model_path, d_name))