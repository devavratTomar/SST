import torch
import torch.nn as nn

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def gramm_matrix(self, x):
        batch = x.shape[0]
        dim_features = x.shape[1]
        features = x.view(batch, dim_features, -1)

        g = torch.bmm(features, features.permute(0, 2, 1)) # gram product
        return g.div(features.shape[1] * features.shape[2])

    def forward(self, input_features, target_features):
        g_input = self.gramm_matrix(input_features)
        g_target = self.gramm_matrix(target_features).detach()
        return nn.functional.l1_loss(g_input, g_target)