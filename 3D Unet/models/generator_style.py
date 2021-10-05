import torch
import torch.nn as nn

from .encoder import Encoder
from .flow_model import FlowGenerator
from .appearance_model import ApperanceModelStyle

class StyleGenerator(nn.Module):
    def __init__(self, nz_dim, style_code_dim, naf):
        super(StyleGenerator, self).__init__()

        self.style_mapper = nn.Sequential(
            nn.Linear(nz_dim, style_code_dim),
            nn.ReLU(True),
            nn.Linear(style_code_dim, style_code_dim),
            nn.ReLU(True),
            nn.Linear(style_code_dim, style_code_dim)
        )

        self.flow_model    = FlowGenerator(nz_dim)
        self.app_model     = ApperanceModelStyle(naf, style_code_dim)


    def forward(self, in_img, in_label, nz):
        style = self.style_mapper(nz)

        # batch_size x style_dim.  Reshape it to batch_size x style_dim, 1, 1, 1
        nz = nz[..., None, None, None]

        # apply flow distortion to the base image and segmentation
        [wrapped_img, wrapped_label] , flow_field =  self.flow_model(in_img, in_label, nz)
        
        predicted_image = self.app_model(wrapped_img, style)

        return predicted_image, wrapped_img, wrapped_label, flow_field