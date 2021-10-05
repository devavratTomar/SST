from models.encoder import EncoderStyle
from models import DisLatentCode, UnetFlowModel

from utilities.util import load_network
import torch
import torch.nn as nn
import os

from loss import GANLoss

class StyleEncoderTrainer(object):
    def __init__(self, opt, K=1024, m=0.99, t=0.07):
        self.opt = opt

        # number of keys, momentum and temperature
        self.K = K
        self.m = m
        self.T = t

        self.style_encoder_q = EncoderStyle(opt.style_code_dim).cuda()
        self.style_encoder_k = EncoderStyle(opt.style_code_dim).cuda()

        self.latent_dis      = DisLatentCode(opt.style_code_dim, 5).cuda()

        # for generating same style images
        self.TransformerModel = UnetFlowModel(opt.out_shape).cuda()
        self.TransformerModel.eval()
        for param in self.TransformerModel.parameters():
            param.requires_grad = False
        

        # load pre-train spatial transformer
        self.TransformerModel = load_network(self.TransformerModel, 'UnetFlowModel', 'latest', opt.checkpoints_dir_pretrained)

        if opt.continue_train:
            # if continue train, load the saved models
            self.style_encoder_q = load_network(self.style_encoder_q, 'Encoder_style_q', 'latest', opt.checkpoints_dir_pretrained)
            self.style_encoder_k = load_network(self.style_encoder_k, 'Encoder_style_k', 'latest', opt.checkpoints_dir_pretrained)
            self.latent_dis      = load_network(self.latent_dis, 'Discriminator_style', 'latest', opt.checkpoints_dir_pretrained)

            self.queue = torch.load(os.path.join(opt.checkpoints_dir_pretrained, 'models', 'queue_' + 'latest.pt'))
        else:
            # create queue to keep keys
            self.queue = torch.randn(opt.style_code_dim, self.K)
            self.queue = nn.functional.normalize(self.queue, dim=0)
            # initialize
            for param_q, param_k in zip(self.style_encoder_q.parameters(), self.style_encoder_k.parameters()):
                param_k.data.copy_(param_q.data)

        # no gradients for encoder_k
        for param_k in self.style_encoder_k.parameters():
            param_k.requires_grad = False

        if len(opt.gpu_ids) > 0:
            self.style_encoder_q = nn.DataParallel(self.style_encoder_q, device_ids=opt.gpu_ids)
            self.style_encoder_k = nn.DataParallel(self.style_encoder_k, device_ids=opt.gpu_ids)
            self.latent_dis      = nn.DataParallel(self.latent_dis, device_ids=opt.gpu_ids)
            self.TransformerModel = nn.DataParallel(self.TransformerModel, device_ids=opt.gpu_ids)

        # queue pointer
        self.queue_ptr = torch.zeros(1, dtype=torch.long)
        
        # create optimizer
        self.optimizer = torch.optim.Adam(self.style_encoder_q.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2), weight_decay=self.opt.weight_decay)


        # separate optimizer for the discriminator
        self.optimizer_dis = torch.optim.Adam(self.latent_dis.parameters(), lr=self.opt.lr, betas=(0.5, 0.999), weight_decay=self.opt.weight_decay)

        self.criterion = nn.CrossEntropyLoss()
        self.criterion_gan = GANLoss('lsgan')

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.style_encoder_q.parameters(), self.style_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = keys.cpu()
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def style_transform(self, img_q, img_k):
        q_styled_img_k, _, _ = self.TransformerModel(img_q, torch.zeros_like(img_q).cuda(), img_k)

        return q_styled_img_k

    def run_step(self, img_q, img_k):
        # first thing first make gradients zero
        self.optimizer.zero_grad()
        img_k = self.style_transform(img_q, img_k)

        # encode img_q and img_k
        q = self.style_encoder_q(img_q) # N x style_dim

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.style_encoder_k(img_k)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().to(device=q.device).detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        # now contrastive loss
        loss_contrast = self.criterion(logits, labels).mean()

        # loss discriminative - if predict of q is 1 loss is low
        loss_gen = self.opt.lambda_gan*self.criterion_gan(self.latent_dis(q), True)

        loss = loss_contrast + loss_gen

        # backpropagate
        loss.backward()
        self.optimizer.step()

        # run discriminator step
        self.optimizer_dis.zero_grad()
        real_latent = torch.randn_like(q).cuda()
        real_latent = nn.functional.normalize(real_latent, dim=1)

        # no need of q
        q = q.detach()
        loss_dis = self.opt.lambda_gan*(self.criterion_gan(self.latent_dis(real_latent), True) +\
                                        self.criterion_gan(self.latent_dis(q), False))
        loss_dis.backward()
        self.optimizer_dis.step()


        ## lossses to monitor
        self.loss = {
            "g_cross_entropy":loss_contrast.detach(),
            "g_adv":loss_gen.detach(),
            "g_all":loss.detach(),
            "d_all":loss_dis.detach()
        }
        self.imgk = img_k.detach()

    def get_latest_losses(self):
        return self.loss

    def update_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, epoch):
        q_model_name = '%s_net_%s.pth' % (epoch, 'Encoder_style_q')
        k_model_name = '%s_net_%s.pth' % (epoch, 'Encoder_style_k')

        d_model_name = '%s_net_%s.pth' % (epoch, 'Discriminator_style')

        queue_name = 'queue_%s.pt' % (epoch)

        saved_model_path = os.path.join(self.opt.checkpoints_dir_pretrained, 'models')

        torch.save(self.style_encoder_q.module.state_dict(), os.path.join(saved_model_path, q_model_name))
        torch.save(self.style_encoder_k.module.state_dict(), os.path.join(saved_model_path, k_model_name))
        torch.save(self.latent_dis.module.state_dict(), os.path.join(saved_model_path, d_model_name))
        torch.save(self.queue, os.path.join(saved_model_path, queue_name))