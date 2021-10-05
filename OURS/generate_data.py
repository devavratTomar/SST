from math import atan
import torch
import torch.nn as nn

import os

from models import UnetFlowModel

import numpy as np
from utilities.util import load_network, load_numpy, save_numpy
from PIL import Image
from evaluations import Constants
K = Constants()


def perform_data_augment_images(rootdir, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    all_images = [f for f in os.listdir(rootdir) if f.endswith('MR.npy.gz')]
    
    for t in K.brain_tests:
        all_images.remove(t + '_MR.npy.gz')
    
    print('size of train aug is ', len(all_images))
    counter = 0

    flow_model = UnetFlowModel([128, 160, 160])
    flow_model_name = 'UnetFlowModel'
 
    flow_model = load_network(flow_model, flow_model_name, 'latest', './checkpoints_pretrained')

    flow_model = flow_model.cuda().eval()

    with torch.no_grad():
        for base_path in all_images:
            base_img = load_numpy(os.path.join(rootdir, base_path)).transpose(2, 1, 0)
            base_seg = load_numpy(os.path.join(rootdir, base_path.replace('MR', 'MR_seg'))).transpose(2, 1, 0)
            
            base_img = torch.from_numpy(base_img).to(torch.float32).cuda()
            base_img = base_img.unsqueeze(0).unsqueeze(0) # add batch and channel
            base_img = 2*base_img - 1.0

            base_seg = torch.from_numpy(base_seg).to(torch.float32).cuda()
            base_seg = base_seg.unsqueeze(0).unsqueeze(0) # add batch and channel

            # create folder if not exists
            if not os.path.exists(os.path.join(outdir, base_path[:-7])):
                os.mkdir(os.path.join(outdir, base_path[:-7]))

            for fixed_path in all_images:
                
                
                if fixed_path == base_path:
                    img_np_base_preview = 255*(0.5*base_img.cpu().numpy()[0, 0, 64] + 0.5)
                    img_np_base_preview = img_np_base_preview.astype(np.uint8)
                    Image.fromarray(img_np_base_preview).save(os.path.join(outdir, base_path[:-7], 'base_atlas.png'))
                    save_numpy(os.path.join(outdir, base_path[:-7], 'base_atlas.npy.gz'), base_img.cpu().numpy()[0, 0])
                    save_numpy(os.path.join(outdir, base_path[:-7], 'base_seg.npy.gz'), base_seg.cpu().numpy()[0, 0])
                    continue

                counter += 1
                save_path = os.path.join(outdir, base_path[:-7], str(counter).zfill(6))
                fixed_img = load_numpy(os.path.join(rootdir, fixed_path)).transpose(2, 1, 0)
                fixed_img = torch.from_numpy(fixed_img).to(torch.float32).cuda()
                fixed_img = fixed_img.unsqueeze(0).unsqueeze(0) # add batch and channel
                fixed_img = 2*fixed_img - 1.0

                predicted_img, predicted_seg, _ = flow_model(base_img, base_seg, fixed_img)
                
                preview = 0.5*predicted_img[0, 0, 64] + 0.5
                preview = 255*preview.cpu().numpy()
                preview = preview.astype(np.uint8)
                Image.fromarray(preview).save(save_path + '.png')

                predicted_img = predicted_img.cpu().numpy()[0, 0]
                predicted_seg = predicted_seg.cpu().numpy()[0, 0]
                save_numpy(save_path + 'atlas.npy.gz', predicted_img)
                save_numpy(save_path + 'seg.npy.gz', predicted_seg)


perform_data_augment_images('../CANDIShare_clean_gz/', '../CANDIShare_StyleAug_gz/')


                
