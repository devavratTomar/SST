import torch
import torch.nn as nn
import os

from models import SpatialTransformer, ApperanceModelStyle, FlowEncoder, FlowGenerator

import numpy as np
from utilities.util import load_network, load_numpy, save_numpy
from PIL import Image
from evaluations import Constants

import random

K = Constants()

def convert_to_image(tensor):
    img = tensor[64]
    img = img
    img = 255.0*img
    img = img.astype(np.uint8)
    return Image.fromarray(img)

def color_seg(tensor):
    img = tensor[64]
    img = img.astype(np.int)

    seg = K.colors[img].astype(np.uint8)
    return Image.fromarray(seg)

def get_noise(batch_size=1, size=128):
    noise = torch.randn(batch_size, size)
    noise = nn.functional.normalize(noise, dim=1)

    return noise

def get_flow_interpolate(flow1, flow2, flow3):
    m3 = 0.3*random.random()
    m2 = 0.3*random.random()

    flow = flow1 + m2*flow2 + m3*flow3
    flow = flow/(1.0 + m2+ m3)
    return flow

def get_flow_pairs(list_flow):
    ret = []
    for f in list_flow:
        for g in list_flow:
            ret.append([f, g])
    return ret

@torch.no_grad()
def generate_fake_data(input_dir, output_dir, flow_path, checkpoint_path, checkpoint_path_pretrained, n_images=82*30, base_img_name='1750_MR'): #n_images=82*60 BPDwoPsy_049_MR
    
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, 'previews'))
        os.makedirs(os.path.join(output_dir, 'images'))
    
    app_model = ApperanceModelStyle(16, 128)
    app_model = load_network(app_model, 'Appearance_Model','latest', checkpoint_path)

    app_model = app_model.cuda()
    app_model.eval()

    flow_encoder = FlowEncoder()
    flow_encoder = load_network(flow_encoder, 'FlowEncoder', 'latest', checkpoint_path_pretrained)
    flow_encoder = flow_encoder.cuda()
    flow_encoder.eval()

    flow_generator = FlowGenerator()
    flow_generator = load_network(flow_generator, 'FlowGenerator', 'latest', checkpoint_path_pretrained)
    flow_generator = flow_generator.cuda()
    flow_generator.eval()
    

    # Spatial transformer
    spatial_transformer = SpatialTransformer([128, 160, 160]).cuda()

    # load the base image and labels
    base_img = load_numpy(os.path.join(input_dir, base_img_name + '.npy.gz')).astype(np.float32).transpose(2, 1, 0)
    base_label = np.rint(load_numpy(os.path.join(input_dir, base_img_name + '_seg.npy.gz'))).astype(np.float32).transpose(2, 1, 0)

    base_img = torch.from_numpy(base_img).unsqueeze(0).unsqueeze(0)
    base_label = torch.from_numpy(base_label).unsqueeze(0).unsqueeze(0)
    
    # normalize them
    base_img = 2.0*base_img.cuda() - 1.0
    base_label = base_label.cuda()

    # all flows
    all_flows = [f for f in os.listdir(flow_path) if f.endswith('.npy.gz')]
    n_flows = len(all_flows)
    print(n_flows)
    n_generated = 0
    upper_loop = n_images//n_flows

    for j in range(upper_loop):
        for path in all_flows:
            n_generated += 1
            print('generating ', n_generated)
            p1, p2, p3 = path, all_flows[random.randint(0, n_flows-1)], all_flows[random.randint(0, n_flows-1)]
            flow1, flow2, flow3 = load_numpy(os.path.join(flow_path, p1)), load_numpy(os.path.join(flow_path, p2)), load_numpy(os.path.join(flow_path, p3))

            flow1 = torch.from_numpy(flow1).cuda()
            flow2 = torch.from_numpy(flow2).cuda()
            flow3 = torch.from_numpy(flow3).cuda()

            flow1 = flow_encoder(torch.from_numpy(flow1).unsqueeze(0))
            flow2 = flow_encoder(torch.from_numpy(flow2).unsqueeze(0))
            flow3 = flow_encoder(torch.from_numpy(flow3).unsqueeze(0))

            flow = get_flow_interpolate(flow1, flow2, flow3)

            flow = flow_generator(flow)

            random_style = get_noise().cuda()
            styled_base_img = app_model(base_img, random_style).detach()

            predicted_img, predicted_label = spatial_transformer(styled_base_img, base_label, flow)
            
            predicted_img = 0.5*predicted_img[0, 0] + 0.5
            predicted_img = predicted_img.cpu().numpy()
            predicted_label = predicted_label[0, 0].cpu().numpy()

            predicted_label = predicted_label*(predicted_img>=0.01)

            save_numpy(os.path.join(output_dir, 'images', str(n_generated) + 'new_atlas.npy.gz'), predicted_img)
            save_numpy(os.path.join(output_dir, 'images', str(n_generated) + 'new_seg.npy.gz'), predicted_label)

            # save previews
            convert_to_image(predicted_img).save(os.path.join(output_dir, 'previews', str(n_generated) + 'new_img.png'))
            color_seg(predicted_label).save(os.path.join(output_dir, 'previews', str(n_generated) + 'new_seg.png'))


generate_fake_data('../../one_shot/CANDIShare_clean_gz',
                   '../Fake_data_new_version/Fake_data_candi',
                   '../FlowFields_new_version/FlowFields_candi',
                   './candi_checkpoints',
                   './candi_checkpoints_pretrained',
                   base_img_name='BPDwoPsy_049_MR')
