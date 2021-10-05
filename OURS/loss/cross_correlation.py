import torch
import torch.nn as nn
import torch.nn.functional as nnf

class CrossCorrelationLoss(nn.Module):
    def __init__(self, win_size=9, use_gpu=True):
        super(CrossCorrelationLoss, self).__init__()
        
        # set window size
        win = [win_size] * 3

        # compute filters
        self.sum_filt = torch.ones([1, 1, *win])

        if use_gpu:
            self.sum_filt = self.sum_filt.to("cuda")
        
        self.pad = win_size//2
        self.win_size = win_size**3

    def forward(self, predicted, target):
        # squares of correlations
        I = predicted
        J = target

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum  =  nnf.conv3d(I, self.sum_filt,  padding=self.pad)
        J_sum  =  nnf.conv3d(J, self.sum_filt,  padding=self.pad)
        I2_sum =  nnf.conv3d(I2, self.sum_filt, padding=self.pad)
        J2_sum =  nnf.conv3d(J2, self.sum_filt,  padding=self.pad)
        IJ_sum =  nnf.conv3d(IJ, self.sum_filt,  padding=self.pad)

        u_I = I_sum / self.win_size
        u_J = J_sum / self.win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * self.win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return torch.mean(cc)
