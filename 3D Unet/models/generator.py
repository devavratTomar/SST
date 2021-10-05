import torch
import torch.nn as nn

from models import ApperanceModel, FlowGenerator


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.flow_model = FlowGenerator(opt.dim, opt.flow_code_dim, opt.nff, opt.out_shape)
        self.appearance_model = ApperanceModel(opt.naf)

    def forward(self, in_img, in_label, nz_flow):
        # change the shape of structures
        (fake_sp_tf_img, fake_label), flow = self.flow_model(in_img, in_label, nz_flow)

        # change appearance
        fake_img = self.appearance_model(fake_sp_tf_img)

        return fake_img, fake_label, flow
