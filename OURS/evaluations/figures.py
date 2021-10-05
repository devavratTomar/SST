import torch
from models import ApperanceModelStyle, EncoderStyle, UnetFlowModel

import os
import numpy as np
from .helpers import *

from utilities.util import load_network, load_numpy


@torch.no_grad()
def generate_main_figure(input_dir, output_dir, checkpoint_dir, color_jitter=True, base_img_path='BPDwoPsy_049_MR'):
    encoder_style_k = EncoderStyle(128)
    encoder_style_k = load_network(encoder_style_k, 'Encoder_style_k', 'latest', checkpoint_dir)
    encoder_style_k.cuda()
    encoder_style_k.eval()

    app_model = ApperanceModelStyle(28, 128)
    app_model = load_network(app_model, 'Appearance_Model', 'latest', checkpoint_dir)
    app_model.cuda()
    app_model.eval()

    flow_model = UnetFlowModel([128, 160, 160])
    flow_model_name = 'UnetFlowModel'
    
    flow_model = load_network(flow_model, flow_model_name, 'latest', checkpoint_dir)
    flow_model.cuda()
    flow_model.eval()


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_imgs = [f for f in os.listdir(input_dir) if f.endswith('MR.npy.gz')]

    all_imgs.remove(base_img_path + '.npy.gz')
    base_img = load_numpy(os.path.join(input_dir, base_img_path + '.npy.gz')).astype(np.float32).transpose(2, 1, 0)
    base_label = load_numpy(os.path.join(input_dir, base_img_path + '_seg.npy.gz')).astype(np.float32).transpose(2, 1, 0)
    
    base_img = torch.from_numpy(base_img).unsqueeze(0).unsqueeze(0)
    base_label = torch.from_numpy(base_label).unsqueeze(0).unsqueeze(0)

    base_img = 2.0*base_img.cuda() - 1.0
    base_label = base_label.cuda()

    # save the base image and base_label

    base_img_png = convert_to_image(base_img)
    base_img_png.save(os.path.join(output_dir, 'base_img.png'))

    base_seg_png = color_seg(base_label)
    base_seg_png.save(os.path.join(output_dir, 'base_img_seg.png'))

    for p in all_imgs:
        fixed_img = load_numpy(os.path.join(input_dir, p)).astype(np.float32).transpose(2, 1, 0)
        fixed_img = torch.from_numpy(fixed_img)

        gt_seg = load_numpy(os.path.join(input_dir, p[:-7] + '_seg.npy.gz')).astype(np.float32).transpose(2, 1, 0)
        gt_seg = torch.from_numpy(gt_seg).unsqueeze(0).unsqueeze(0)

        # change style of fixed img
        if color_jitter:
            fixed_img = random_color_jitter(fixed_img)
        
        fixed_img = fixed_img.unsqueeze(0).unsqueeze(0)
        fixed_img = 2.0*fixed_img.cuda() - 1.0

        # change style of base img
        target_style = encoder_style_k(fixed_img)

        base_with_target_style = app_model(base_img, target_style)

        predicted_img, predicted_label, _ = flow_model(base_with_target_style, base_label, fixed_img)

        # save the predictions
        convert_to_image(predicted_img).save(os.path.join(output_dir, p[:-7] + '_pred_img' + '.png'))
        color_seg(predicted_label).save(os.path.join(output_dir, p[:-7] + '_pred_seg' + '.png'))
        convert_to_image(base_with_target_style).save(os.path.join(output_dir, p[:-7] + '_base_ts' + '.png'))
        convert_to_image(fixed_img).save(os.path.join(output_dir, p[:-7] + '_gt_img' + '.png'))
        color_seg(gt_seg).save(os.path.join(output_dir, p[:-7] + '_gt_seg' + '.png'))
    
if __name__ == '__main__':
    generate_main_figure('../CANDIShare_clean/', '../figures_paper/', './checkpoints')