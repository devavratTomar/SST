from models.appearance_model import ApperanceModelStyle
from models.encoder import EncoderStyle
from utilities.util import load_network
import torch
import torch.nn as nn
import os

from loss import StyleLoss

class AppearanceModelTrainer(object):
    def __init__(self, opt):
        self.opt = opt
        
        self.appearance_model = ApperanceModelStyle(opt.nf, opt.style_code_dim)
        self.style_encoder = EncoderStyle(opt.style_code_dim)
        
        # load the pretrained style_encoder
        self.style_encoder = load_network(self.style_encoder, 'Encoder_style_q', 'latest', opt.checkpoints_dir_pretrained)

        # freeze the weights of style encoder
        for param in self.style_encoder.parameters():
            param.requires_grad = False

        if opt.continue_train:
            self.appearance_model = load_network(self.appearance_model, 'Appearance_Model', 'latest', opt.checkpoints_dir_pretrained)
        
        if len(opt.gpu_ids) > 0:
            self.style_encoder = self.style_encoder.cuda()
            self.appearance_model = self.appearance_model.cuda()

        
        # adam optimizer
        self.optimizer = torch.optim.Adam(self.appearance_model.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2), weight_decay=self.opt.weight_decay)

        # loss
        self.criterianl1 = nn.L1Loss()
        self.criterianstyle = StyleLoss()
    
    def encode_style(self, img):
        q = self.style_encoder(img)
        return q

    def compute_style_loss(self, input, target):
        feat_input = self.style_encoder.backbone(input)
        feat_input = self.style_encoder.conv1x1(feat_input)

        feat_target = self.style_encoder.backbone(target)
        feat_target = self.style_encoder.conv1x1(feat_target)

        return self.criterianstyle(feat_input, feat_target)


    def run_step(self, img_x, img_y):
        self.optimizer.zero_grad()

        # style code of x and y
        x_style = self.encode_style(img_x).detach()
        y_style = self.encode_style(img_y).detach()

        # prediction by appearance model for x and y
        x_with_style_y = self.appearance_model(img_x, y_style)
        y_with_style_x = self.appearance_model(img_y, x_style)

        self.all_losses = {}

        # style consistency
        loss = self.compute_style_loss(x_with_style_y, img_y) + self.compute_style_loss(y_with_style_x, img_x)
        loss = self.opt.lambda_style_consistency*loss
        self.all_losses.update({'style_consistency': loss})

        # style identity loss
        loss = self.criterianl1(self.appearance_model(img_x, x_style), img_x) + self.criterianl1(self.appearance_model(img_y, y_style), img_y)
        loss = self.opt.lambda_style_identity*loss
        self.all_losses.update({'style_identity': loss})
        
        loss = sum(self.all_losses.values()).mean()
        
        loss.backward()
        self.optimizer.step()

        self.x_with_style_y, self.y_with_style_x = x_with_style_y.detach(), y_with_style_x.detach()

    def get_latest_losses(self):
        return self.all_losses

    def update_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, epoch):
        model_name = '%s_net_%s.pth' % (epoch, 'Appearance_Model')
        saved_model_path = os.path.join(self.opt.checkpoints_dir_pretrained, 'models')
        torch.save(self.appearance_model.state_dict(), os.path.join(saved_model_path, model_name))