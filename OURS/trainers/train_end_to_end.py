import torch
import torch.nn as nn
import os

from utilities.util import load_network

from models import ApperanceModelStyle, EncoderStyle, UnetFlowModel, DisLatentCode

from loss import StyleLoss, GradientLoss, GradientLossImage, CrossCorrelationLoss, GANLoss
from pytorch_msssim import MS_SSIM


class AllTrainer(object):
    def __init__(self, opt):
        self.opt = opt

        # Flow encoder
        self.FlowModel = UnetFlowModel(opt.out_shape).cuda()
        self.FlowModel_name = 'UnetFlowModel'


        # Style encoder: query and key
        self.Style_Encoder_q = EncoderStyle(opt.style_code_dim).cuda()
        self.style_queue     = nn.functional.normalize(torch.randn(opt.style_code_dim, opt.n_style_keys, device='cpu'), dim=0)
        
        # queue pointer
        self.style_queue_ptr = torch.zeros(1, dtype=torch.long)
        
        self.Style_Encoder_k = EncoderStyle(opt.style_code_dim).cuda()


        # transformer model
        # for generating same style images
        self.TransformerModel = UnetFlowModel(opt.out_shape).cuda()
        self.TransformerModel.eval()
        for param in self.TransformerModel.parameters():
            param.requires_grad = False

        # load pre-train spatial transformer
        self.TransformerModel = load_network(self.TransformerModel, 'UnetFlowModel', 'latest', opt.checkpoints_dir_pretrained)
        
        # initialize
        for param_q, param_k in zip(self.Style_Encoder_q.parameters(), self.Style_Encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
        
        # Appearance model
        self.App_Model = ApperanceModelStyle(opt.nf, opt.style_code_dim).cuda()

        self.latent_style_d = DisLatentCode(opt.style_code_dim, 5).cuda()

        # Use pre-trained style?
        if opt.use_pretrain:

            # style encoder query and key models
            self.Style_Encoder_q = load_network(self.Style_Encoder_q, 'Encoder_style_q', 'latest', opt.checkpoints_dir_pretrained)
            self.Style_Encoder_k = load_network(self.Style_Encoder_k, 'Encoder_style_k', 'latest', opt.checkpoints_dir_pretrained)
            self.style_queue     = torch.load(os.path.join(opt.checkpoints_dir_pretrained, 'models', 'queue_' + 'latest.pt'))
            self.latent_style_d = load_network(self.latent_style_d, 'Discriminator_style', 'latest', opt.checkpoints_dir_pretrained)
        
        # continue train?
        if opt.continue_train:
            # spatial transformer encoder-decoder
            self.FlowModel          = load_network(self.FlowModel, self.FlowModel_name, 'latest', opt.checkpoints_dir)

            # style encoder query and key models
            self.Style_Encoder_q = load_network(self.Style_Encoder_q, 'Encoder_style_q', 'latest', opt.checkpoints_dir)
            self.style_queue     = torch.load(os.path.join(opt.checkpoints_dir, 'models', 'queue_' + 'latest.pt'))

            self.Style_Encoder_k = load_network(self.Style_Encoder_k, 'Encoder_style_k', 'latest', opt.checkpoints_dir)
            # appearance model
            self.App_Model       = load_network(self.App_Model, 'Appearance_Model', 'latest', opt.checkpoints_dir)

            self.latent_style_d  = load_network(self.latent_style_d, 'Discriminator_style', 'latest', opt.checkpoints_dir)

        # no gradient for the key encoder model
        for param in self.Style_Encoder_k.parameters():
            param.requires_grad = False

        # utilize multiple gpus
        if len(opt.gpu_ids) > 0:
            self.FlowModel = nn.DataParallel(self.FlowModel, device_ids=opt.gpu_ids)
            self.Style_Encoder_q = nn.DataParallel(self.Style_Encoder_q, device_ids=opt.gpu_ids)
            
            self.Style_Encoder_k = nn.DataParallel(self.Style_Encoder_k, device_ids=opt.gpu_ids)
            
            self.App_Model = nn.DataParallel(self.App_Model, device_ids=opt.gpu_ids)
            self.latent_style_d = nn.DataParallel(self.latent_style_d, device_ids=opt.gpu_ids)

            self.TransformerModel = nn.DataParallel(self.TransformerModel, device_ids=opt.gpu_ids)
        
        # optimizer: Adam
        params = list(self.FlowModel.parameters()) + list(self.App_Model.parameters()) + list(self.Style_Encoder_q.parameters())
        
        self.optimizer = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))

        # discriminator optimizer
        self.optimizer_dis = torch.optim.Adam(self.latent_style_d.parameters(), lr=self.opt.lr, betas=(0.5, 0.999),weight_decay=self.opt.weight_decay)

        # losses
        self.l1_loss            = nn.L1Loss()
        self.style_loss         = StyleLoss()
        self.grad_loss          = GradientLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.grad_img_loss      = GradientLossImage()
        self.gan_loss           = GANLoss('lsgan')
        self.ssim_loss          = MS_SSIM(1.0, win_size=7, channel=1, spatial_dims=3)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.Style_Encoder_q.parameters(), self.Style_Encoder_k.parameters()):
            param_k.data = param_k.data * 0.99 + param_q.data * (1. - 0.99)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = keys.cpu()
        batch_size = keys.shape[0]

        ptr = int(self.style_queue_ptr)
        assert self.opt.n_style_keys % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.style_queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.opt.n_style_keys  # move pointer

        self.style_queue_ptr[0] = ptr
    
    @torch.no_grad()
    def style_transform(self, img_q, img_k):
        q_styled_img_k, _, _ = self.TransformerModel(img_q, torch.zeros_like(img_q).cuda(), img_k)

        return q_styled_img_k
    
    def moco(self, img_q, img_k):
        # transfer style of q to img_k
        img_k = self.style_transform(img_q, img_k)
        # encode image query and key
        q = self.Style_Encoder_q(img_q)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.Style_Encoder_k(img_k)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.style_queue.clone().to(device=q.device).detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= 0.07

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        # now contrastive loss
        return self.cross_entropy_loss(logits, labels).mean()
    
    def compute_style_loss(self, input, target):
        feat_input = self.Style_Encoder_k.module.backbone(input)
        feat_input = self.Style_Encoder_k.module.conv1x1(feat_input)

        feat_target = self.Style_Encoder_k.module.backbone(target)
        feat_target = self.Style_Encoder_k.module.conv1x1(feat_target)

        return self.style_loss(feat_input, feat_target)

    def compute_imgdiff_loss(self, input_img, target_img, alpha=0.5):
        return (1.0 - alpha)*self.l1_loss(input_img, target_img) + alpha*(1.0 - self.ssim_loss(0.5*input_img + 0.5, 0.5*target_img + 0.5))


    @torch.no_grad()
    def run_test(self, base_img, base_label, target_img):
        target_style = self.Style_Encoder_k(target_img)
        target_styled_base_img = self.App_Model(base_img, target_style)
        predicted_img, predicted_label, _ = self.FlowModel(target_styled_base_img, base_label, target_img)

        return target_styled_base_img, predicted_img, predicted_label

    def run_step_no_base(self, base_img, base_label, source_img, target_img):
        """
        Instead of feeding the same base imgage to appearance model we train it on all images so that it does not overfit
        """
        self.optimizer.zero_grad()
        dummy_labels = torch.zeros_like(base_label)

        source_style = self.Style_Encoder_k(source_img)
        target_style = self.Style_Encoder_k(target_img)

        target_styled_source_img = self.App_Model(source_img, target_style)

        # we spatially transform (warp) source image to target_img
        predicted_target_img, _, feild_source2target = self.FlowModel(target_styled_source_img, dummy_labels, target_img)
  
        # losses
        loss_grad = self.grad_loss(feild_source2target)

        # prediction l1 and ssim loss
        loss_img_diff = self.compute_imgdiff_loss(predicted_target_img, target_img)

        # style consistency loss
        source_img_style_recon = self.App_Model(target_styled_source_img, source_style)
  
        # style consistency loss
        loss_style_consistency = self.compute_imgdiff_loss(source_img_style_recon, source_img)

        # # style identity loss
        target_img_idt = self.App_Model(target_img, target_style)
        loss_style_idt = self.compute_imgdiff_loss(target_img_idt, target_img)

        # moco loss
        loss_moco = self.moco(target_img, source_img)

        # style adversarial loss
        target_style_q = self.Style_Encoder_q(target_img)
        loss_adv_g = self.opt.lambda_gan*self.gan_loss(self.latent_style_d(target_style_q), True)


        loss = loss_moco + self.opt.final_lambda_1*(loss_style_consistency + loss_style_idt) \
            + self.opt.final_lambda_2*(loss_img_diff + self.opt.final_lambda_reg*loss_grad) + self.opt.final_lambda_3*loss_adv_g

        loss.backward()
        self.optimizer.step()

        # discriminator
        self.optimizer_dis.zero_grad()
        real_latent = torch.randn_like(target_style).cuda()
        real_latent = nn.functional.normalize(real_latent, dim=1)
        
        # no need of q
        target_style_q = target_style_q.detach()
        loss_dis = 0.5*(self.gan_loss(self.latent_style_d(real_latent), True) + self.gan_loss(self.latent_style_d(target_style_q), False))
        
        loss_dis_scaled = self.opt.final_lambda_3*loss_dis
        loss_dis_scaled.backward()
        self.optimizer_dis.step()

        # for visualization only
        self.all_losses = {
            'moco':loss_moco.detach(),
            'grad': loss_grad.detach(),
            'img_diff': loss_img_diff.detach(),
            'style_consistency':loss_style_consistency.detach(),
            'loss_style_idt':loss_style_idt.detach(),
            'loss_style_g':loss_adv_g.detach(),
            'loss_style_d':loss_dis.detach(),
            'loss':loss.detach()
        }

        self.base_img_with_target_style, self.predicted_img, self.predicted_label =  self.run_test(base_img, base_label, target_img)
    
   
    def get_latest_losses(self):
        return self.all_losses

    def update_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, epoch):
        f_name  = '%s_net_%s.pth' % (epoch, self.FlowModel_name)
        se_q    = '%s_net_%s.pth' % (epoch, 'Encoder_style_q')
        se_k    = '%s_net_%s.pth' % (epoch, 'Encoder_style_k')
        am      = '%s_net_%s.pth' % (epoch, 'Appearance_Model')
        l_d     = '%s_net_%s.pth' % (epoch, 'Discriminator_style')
        queue_name = 'queue_%s.pt' % (epoch)

        saved_model_path = os.path.join(self.opt.checkpoints_dir, 'models')
        
        torch.save(self.FlowModel.module.state_dict(), os.path.join(saved_model_path, f_name))

        torch.save(self.Style_Encoder_q.module.state_dict(), os.path.join(saved_model_path, se_q))
        torch.save(self.style_queue, os.path.join(saved_model_path, queue_name))
        
        torch.save(self.Style_Encoder_k.module.state_dict(), os.path.join(saved_model_path, se_k))
        torch.save(self.App_Model.module.state_dict(), os.path.join(saved_model_path, am))
        torch.save(self.latent_style_d.module.state_dict(), os.path.join(saved_model_path, l_d))
