from models import ApperanceModelStyle, FlowGenerator, SpatialTransformer
import time
import torch

import os


os.environ["CUDA_VISIBLE_DEVICES"]='1'

def time_generation(device):
    app_model = ApperanceModelStyle(16, 128).to(device=device)
    flow_decoder = FlowGenerator().to(device=device)
    spatial_transformer = SpatialTransformer([128, 160, 160]).to(device=device)

    if device == 'cpu':
        start_time = time.time()
        for i in range(100):
            test_img = torch.randn(1, 1, 128, 160, 160).to(device=device)
            styled_img = app_model(test_img, torch.rand(1, 128).to(device=device)).detach()
            flow_rand = flow_decoder(torch.randn(1, 64, 4, 5, 5).to(device=device))

            img_rand, seg_rand = spatial_transformer(styled_img, styled_img, flow_rand)
        end_time = time.time()
        time_diff = end_time - start_time

    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        for i in range(10):
            test_img = torch.randn(1, 1, 128, 160, 160).to(device=device)
            styled_img = app_model(test_img, torch.rand(1, 128).to(device=device)).detach()
            flow_rand = flow_decoder(torch.randn(1, 64, 4, 5, 5).to(device=device))

            img_rand, seg_rand = spatial_transformer(styled_img, styled_img, flow_rand)
        
        end.record()
        
        # Waits for everything to finish running
        torch.cuda.synchronize()
        time_diff = start.elapsed_time(end)
    
    print("Took --- ", time_diff, " seconds --- on ", device)


# time_generation('cpu')
time_generation('cuda:0')