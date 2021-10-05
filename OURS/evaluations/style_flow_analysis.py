import torch
import torch.nn.functional as nnf

from models import ApperanceModelStyle, SpatialTransformer, FlowGenerator, FlowEncoder

from utilities.util import load_network, load_numpy
import random
from PIL import Image
import numpy as np
import os
import random

from .constants import Constants
K = Constants()

def load_imgs(paths):
    imgs = []

    for p in paths:
        img = load_numpy(p).astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)
        imgs.append(img)
    
    out = torch.cat(imgs, dim=0)
    return out

def save_img_batch(imgs, outfolder):
    imgs = 0.5*imgs.cpu().numpy() + 0.5
    imgs = 255*imgs
    imgs = imgs.astype(np.uint8)

    for i, img in enumerate(imgs):
        Image.fromarray(img[0][64]).save(os.path.join(outfolder, str(i)+'.png'))

def color_seg(tensor):
    seg = K.colors[tensor].astype(np.uint8)
    return seg

@torch.no_grad()
def generate_some_styles_flows(input_dir, flow_dir, output_path, checkpoints_path , base_img_path='BPDwoPsy_049_MR', nstyles=4, nflows=4, n_experiments=10):
    if not os.path.exists(output_path):
        os.makedirs(os.path.join(output_path))

    app_model = ApperanceModelStyle(16, 128)
    app_model = load_network(app_model, 'Appearance_Model', '1', checkpoints_path)
    app_model.cuda()
    app_model.eval()

    spatial_transformer = SpatialTransformer([128, 160, 160]).cuda()
    # load the base image and labels
    base_img = load_numpy(os.path.join(input_dir, base_img_path + '.npy.gz')).astype(np.float32).transpose(2, 1, 0)
    base_label = np.rint(load_numpy(os.path.join(input_dir, base_img_path + '_seg.npy.gz'))).astype(np.float32).transpose(2, 1, 0)

    base_img = torch.from_numpy(base_img).unsqueeze(0).unsqueeze(0)
    base_label = torch.from_numpy(base_label).unsqueeze(0).unsqueeze(0)

    base_img = 2.0*base_img.cuda() - 1.0
    base_label = base_label.cuda()

    all_flows = [os.path.join(flow_dir, f) for f in os.listdir(flow_dir) if f.endswith('npy.gz')]

    for i in range(n_experiments):
        if not os.path.exists(os.path.join(output_path, 'experiment_' + str(i))):
            os.makedirs(os.path.join(output_path, 'experiment_' + str(i)))
        
        # sample 4 random styles
        random_styles = nnf.normalize(torch.randn(nstyles, 128).cuda(), dim=1)
        #base_img      = base_img.repeat(nstyles, 1, 1, 1, 1)
        random_base_img = app_model(base_img, random_styles)

        # sample 4 random flows
        random_flow_paths = random.sample(all_flows, nflows)
        target_flows = load_imgs(random_flow_paths).cuda()

        for j, target in enumerate(target_flows):
            target = target.repeat(nstyles, 1, 1, 1, 1)
            predicted_imgs, _ = spatial_transformer(random_base_img, torch.randn_like(random_base_img).cuda(), target)
            if not os.path.exists(os.path.join(output_path, 'experiment_' + str(i), 'target_' + str(j))):
                os.makedirs(os.path.join(output_path, 'experiment_' + str(i), 'target_' + str(j)))
            save_img_batch(predicted_imgs, os.path.join(output_path, 'experiment_' + str(i), 'target_' + str(j)))
            


@torch.no_grad()
def flow_latent_walk(input_dir, output_path, flow_dir, checkpoints_path, checkpoints_path_flow, base_img_path='BPDwoPsy_049_MR', nstyles=4, nflows=10, n_experiments=10):
    if not os.path.exists(output_path):
        os.makedirs(os.path.join(output_path))

    app_model = ApperanceModelStyle(16, 128)
    app_model = load_network(app_model, 'Appearance_Model', '1', checkpoints_path)
    app_model = app_model.cuda()
    app_model.eval()


    flow_generator = FlowGenerator().cuda()
    flow_generator = load_network(flow_generator, 'FlowGenerator', 'latest', checkpoints_path_flow)
    flow_generator.eval()

    flow_encoder = FlowEncoder().cuda()
    flow_encoder = load_network(flow_encoder, 'FlowEncoder', 'latest', checkpoints_path_flow)
    flow_encoder.eval()

    spatial_transformer = SpatialTransformer([128, 160, 160]).cuda()
    
    # load the base image and labels
    base_img = load_numpy(os.path.join(input_dir, base_img_path + '.npy.gz')).astype(np.float32).transpose(2, 1, 0)
    base_label = np.rint(load_numpy(os.path.join(input_dir, base_img_path + '_seg.npy.gz'))).astype(np.float32).transpose(2, 1, 0)

    base_img = torch.from_numpy(base_img).unsqueeze(0).unsqueeze(0)
    base_label = torch.from_numpy(base_label).unsqueeze(0).unsqueeze(0)

    base_img = 2.0*base_img.cuda() - 1.0
    base_label = base_label.cuda()

    # nstyles styles
    style_codes = nnf.normalize(torch.randn(nstyles, 128).cuda(), dim=1)

    final_imgs = []
    final_labels = []

    styled_imgs = app_model(base_img.repeat(nstyles, 1, 1, 1, 1), style_codes)

    all_flows = [os.path.join(flow_dir, f) for f in os.listdir(flow_dir) if f.endswith('npy.gz')]
    random.shuffle(all_flows)
    all_flows = load_imgs(all_flows[:2*nstyles]).cuda()
    all_flow_codes = flow_encoder(all_flows)

    flow_codes_init = all_flow_codes[:nstyles]
    flow_codes_end = all_flow_codes[nstyles:]



    # linear interpolation of flow_codes (lets take 10 points between init and end)
    scales = np.linspace(0.0, 1.0, nflows)
    flow_codes_linear = [(1.0-scale)*flow_codes_init + scale*flow_codes_end for scale in scales]
    flow_fields_linear = [flow_generator(f) for f in flow_codes_linear]

    for flow_field_linear in flow_fields_linear:
        pred_imgs, pred_labels = spatial_transformer(styled_imgs, base_label.repeat(nstyles, 1, 1, 1, 1), flow_field_linear)

        # nstyles, 160, 160
        pred_imgs = 255.0*(0.5*pred_imgs[:, 0, 64, :, :] + 0.5)
        pred_imgs = pred_imgs.cpu().numpy().astype(np.uint8)
        pred_imgs = pred_imgs.reshape(-1, 160)
        final_imgs.append(pred_imgs)

        pred_labels = np.rint(pred_labels[:, 0, 64, :, :].cpu().numpy())
        pred_labels = pred_labels.reshape(-1, 160).astype(int)
        final_labels.append(pred_labels)

    final_imgs = np.concatenate(final_imgs, axis=1)
    final_labels = np.concatenate(final_labels, axis=1)
    final_labels = color_seg(final_labels)

    Image.fromarray(final_labels).save(os.path.join(output_path, 'linear_labels.png'))
    Image.fromarray(final_imgs).save(os.path.join(output_path, 'linear_imgs.png'))

