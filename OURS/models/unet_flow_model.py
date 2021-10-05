from models import Unet, SpatialTransformer

import torch.nn as nn
import torch

class UnetFlowModel(nn.Module):
    def __init__(self, output_shape):
        super(UnetFlowModel, self).__init__()

        self.unet_model = Unet()
        self.spatial_transformer = SpatialTransformer(output_shape)

    def forward(self, moving_img, moving_label, fixed_img):
        x_in = torch.cat([fixed_img, moving_img], axis=1)

        flow = self.unet_model(x_in)

        predicted_img, predicted_label = self.spatial_transformer(moving_img, moving_label, flow)

        return predicted_img, predicted_label, flow


