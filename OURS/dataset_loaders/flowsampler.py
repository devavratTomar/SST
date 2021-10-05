from torch.utils.data import Dataset
from utilities.util import load_numpy
import os
import random
import numpy as np
import torch

class FlowSampler(Dataset):
    def __init__(self, dataroot, img_dir='../CANDIShare_clean_gz', base_img_paths=['BPDwoPsy_049_MR']):
        self.paths_flow = [f for f in os.listdir(dataroot) if f.endswith('npy.gz')]
        self.dataroot = dataroot + '_' + base_img_paths[0]

        self.len = len(self.paths_flow)

        base_img = load_numpy(os.path.join(img_dir, base_img_paths[0] + '.npy.gz'))
        base_label = load_numpy(os.path.join(img_dir, base_img_paths[0] + '_seg.npy.gz'))

        base_img = torch.from_numpy(base_img).to(torch.float32)
        base_img = base_img.permute(2, 1, 0).unsqueeze(0) # add channel

        base_img = 2.0*base_img - 1.0

        base_label = torch.from_numpy(base_label).to(torch.float32)
        base_label = base_label.permute(2, 1, 0).unsqueeze(0) # add channel

        self.base_img = base_img
        self.base_label = base_label
        self.all_flows = [load_numpy(os.path.join(self.dataroot, i)) for i in self.paths_flow]

    def get_flow_interpolate(self, flow1, flow2, flow3):
        m3 = 0.3*random.random()
        m2 = 0.3*random.random()

        flow = flow1 + m2*flow2 + m3*flow3
        flow = flow/(1.0 + m2+ m3)
        return flow

    def __getitem__(self, index):
        flow_1 = self.all_flows[index]

        idx_1, idx_2 = random.randint(0, self.len -1), random.randint(0, self.len -1)
        flow_2 = self.all_flows[idx_1]
        flow_3 = self.all_flows[idx_2]

        flow = self.get_flow_interpolate(flow_1, flow_2, flow_3)

        flow = flow.astype(np.float32)
        flow = torch.from_numpy(flow)
        
        # flow has shape 3, 128, 160, 160
        return flow, self.base_img, self.base_label

    def __len__(self):
        return self.len