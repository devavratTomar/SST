from models import UnetFlowModel

import torch.nn as nn
from utilities.util import load_network
import torch

from loss import CrossCorrelationLoss, GradientLoss
import os

class AETrainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.ae_flow_model = UnetFlowModel(opt.out_shape).cuda()
        self.ae_flow_model_name = 'UnetFlowModel'
        
        if opt.continue_train:
            self.ae_flow_model = load_network(self.ae_flow_model, self.ae_flow_model_name, 'latest', opt.checkpoints_dir_pretrained)
        
        if len(opt.gpu_ids) > 0:
            self.ae_flow_model = nn.DataParallel(self.ae_flow_model, device_ids=opt.gpu_ids)

        self.criterian_cross = CrossCorrelationLoss()
        self.criterian_grad = GradientLoss()
            
        self.optimizer = torch.optim.Adam(list(self.ae_flow_model.parameters()), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2), weight_decay=self.opt.weight_decay)


    def run_step(self, base_img, base_label, real_img):
        self.optimizer.zero_grad()

        predicted_img, predicted_label, flow = self.ae_flow_model(base_img, base_label, real_img)

        ## losses
        loss_ccn = self.opt.lambda_struct*(1.0 - self.criterian_cross((predicted_img + 1.0)/2, (real_img + 1.0)/2))
        loss_grd = self.opt.lambda_grad*self.criterian_grad(flow)

        loss = loss_ccn + loss_grd
        loss.backward()

        self.optimizer.step()
        
        # store output values
        self.predicted_img, self.predicted_label = predicted_img.detach(), predicted_label.detach()
        
        self.all_losses = {
            'grad': loss_grd.detach(),
            'ccn': loss_ccn.detach()
        }

    def get_latest_losses(self):
        return self.all_losses

    def update_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, epoch):
        name = '%s_net_%s.pth' % (epoch, self.ae_flow_model_name)

        saved_model_path = os.path.join(self.opt.checkpoints_dir_pretrained, 'models')
        torch.save(self.ae_flow_model.module.state_dict(), os.path.join(saved_model_path, name))