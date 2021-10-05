import torch
import torch.nn as nn
import os

from models import ApperanceModelStyle, EncoderStyle, UnetFlowModel

import numpy as np
from utilities.util import load_network, load_numpy, save_numpy

from evaluations import Constants

@torch.no_grad()
def generate_flow_fields(input_dir, output_dir, checkpoint_path, epoch='1', base_img_name='BPDwoPsy_049_MR', dataset_name='CANDIShare'):
    output_dir = output_dir + '_' + base_img_name
    K = Constants(dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir))

    app_model = ApperanceModelStyle(16, 128)
    app_model = load_network(app_model, 'Appearance_Model', epoch, checkpoint_path)

    app_model = app_model.cuda()
    app_model.eval()

    style_encoder = EncoderStyle(128)
    style_encoder = load_network(style_encoder, 'Encoder_style_k', epoch, checkpoint_path)

    style_encoder = style_encoder.cuda()
    style_encoder.eval()

    flow_model = UnetFlowModel([128, 160, 160])
    flow_model = load_network(flow_model, 'UnetFlowModel', epoch, checkpoint_path)
    
    flow_model.cuda()
    flow_model.eval()
    
    # load the base image and labels
    base_img = load_numpy(os.path.join(input_dir, base_img_name + '.npy.gz')).astype(np.float32).transpose(2, 1, 0)
    base_label = np.rint(load_numpy(os.path.join(input_dir, base_img_name + '_seg.npy.gz'))).astype(np.float32).transpose(2, 1, 0)

    base_img = torch.from_numpy(base_img).unsqueeze(0).unsqueeze(0)
    base_label = torch.from_numpy(base_label).unsqueeze(0).unsqueeze(0)
    
    # normalize them
    base_img = 2.0*base_img.cuda() - 1.0
    base_label = base_label.cuda()

    all_images = [f for f in os.listdir(input_dir) if f.endswith('MR.npy.gz')]
    all_images.remove(base_img_name + '.npy.gz')

    for test in K.brain_tests:
        all_images.remove(test+'_MR.npy.gz')
    
    counter = 0
    for i, p in enumerate(all_images):
        print('processing ', p)
        counter += 1
        target_img = load_numpy(os.path.join(input_dir, p)).astype(np.float32).transpose(2, 1, 0)
        target_label = np.rint(load_numpy(os.path.join(input_dir, p.replace('MR', 'MR_seg')))).astype(np.float32).transpose(2, 1, 0)

        target_img = torch.from_numpy(target_img).unsqueeze(0).unsqueeze(0)
        target_label = torch.from_numpy(target_label).unsqueeze(0).unsqueeze(0)
        target_img = 2.0*target_img.cuda() -1.0
        
        target_style = style_encoder(target_img)
        base_with_target_style = app_model(base_img, target_style)
        _, _, flow_field = flow_model(base_with_target_style, base_label, target_img)

        flow_field = flow_field[0] # remove batch dim
        flow_field = flow_field.cpu().numpy()

        save_numpy(os.path.join(output_dir, str(counter) + '_flow.npy.gz'), flow_field)

# generate_flow_fields('../CANDIShare_clean_gz', '../FlowFields', './checkpoints')


generate_flow_fields('../OASIS_clean', '../FlowFields_new_version/FlowFields_oasis/', './oasis_checkpoints_new_idea', epoch='latest', base_img_name='1750_MR', dataset_name='oasis')
# generate_flow_fields('../../one_shot/CANDIShare_clean_gz', '../FlowFields_new_version/FlowFields_candi', './candi_checkpoints_new_idea', epoch='latest', base_img_name='BPDwoPsy_049_MR', dataset_name='CANDIShare')