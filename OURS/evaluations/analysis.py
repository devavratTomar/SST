import torch
import os
import numpy as np
from models import UnetFlowModel

from utilities.util import load_network, load_numpy
from PIL import Image

from .constants import Constants

K = Constants()

def generate_img_visuals(volume):
    d, m, n = volume.shape

    visuals = volume[(d//2-8):(d//2+8), ...]
    output = np.zeros((m*4, n*4))
    
    for i in range(4):
        for j in range(4):
            it = 4*i + j
            img = visuals[it]
            img[img > 1.0]

            output[i*m : (i+1)*m, j*n : (j+1)*n] = img

    return output

def generate_seg_visuals(seg_volume):
    seg_volume = seg_volume.astype(np.uint8)
    d, m, n = seg_volume.shape

    visuals = seg_volume[(d//2-8):(d//2+8), ...]
    output = np.zeros((m*4, n*4, 3))

    for i in range(4):
        for j in range(4):
            it = 4*i + j

            img = visuals[it]
            output[i*m : (i+1)*m, j*n : (j+1)*n, :] = K.colors[img]
    
    return output

def get_accuracies(predicted_label, gt_labels):
    pass

def analyse_autoencoder_results(dataroot, outdir, base_file='BPDwoPsy_049_MR'):
    # load base and downsample by 2
    base_anatomy = load_numpy(os.path.join(dataroot, base_file + '.npy.gz')).transpose(2, 1, 0)
    base_seg = load_numpy(os.path.join(dataroot, base_file + '_seg.npy.gz')).transpose(2, 1, 0)

    # pytorch tensor and gpu
    base_anatomy = torch.from_numpy(base_anatomy).to(torch.float32).unsqueeze(0).unsqueeze(0) # add channel and batch axis
    base_seg     = torch.from_numpy(base_seg).to(torch.float32).unsqueeze(0).unsqueeze(0) # add one and batch axis channel to labels

    # transfer to gpu
    base_anatomy = 2*base_anatomy.cuda() - 1.0
    base_seg = base_seg.cuda()


    # all images
    all_anatomies = sorted([f for f in os.listdir(dataroot) if f.endswith('MR.npy.gz') and not base_file in f])
    all_seg       = sorted([f for f in os.listdir(dataroot) if f.endswith('MR_seg.npy.gz') and not base_file in f])

    assert len(all_anatomies) == len(all_seg)



    flow_model = UnetFlowModel([128, 160, 160])
    flow_model_name = 'UnetFlowModel'
    
    flow_model = load_network(flow_model, flow_model_name, 'latest', './checkpoints_pretrained')
    
    flow_model = flow_model.cuda().eval()

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    if not os.path.exists(os.path.join(outdir, 'volume')):
        os.mkdir(os.path.join(outdir, 'volume'))
    
    if not os.path.exists(os.path.join(outdir, 'visuals')):
        os.mkdir(os.path.join(outdir, 'visuals'))
    

    with torch.no_grad():
        for i in range(len(all_anatomies)):
            print('processing %d' % i)
            anatomy = load_numpy(os.path.join(dataroot, all_anatomies[i])).transpose(2, 1, 0)
            np.save(os.path.join(outdir, 'volume', 'gt_' + all_anatomies[i]), anatomy)
            # visuals
            anatomy_visuals = generate_img_visuals(anatomy)
            anatomy_visuals = (255.0*anatomy_visuals).astype(np.uint8)
            Image.fromarray(anatomy_visuals).save(os.path.join(outdir, 'visuals', 'gt_' + all_anatomies[i]+'.png'))
            

            gt_seg = load_numpy(os.path.join(dataroot, all_seg[i])).transpose(2, 1, 0)
            np.save(os.path.join(outdir, 'volume', 'gt_' + all_seg[i]), gt_seg)
            
            # visuals
            gt_seg_visuals = generate_seg_visuals(gt_seg)
            gt_seg_visuals = gt_seg_visuals.astype(np.uint8)
            Image.fromarray(gt_seg_visuals).save(os.path.join(outdir, 'visuals', 'gt_' + all_seg[i]+'.png'))

            # pytorch tensor
            anatomy = torch.from_numpy(anatomy).to(torch.float32).unsqueeze(0).unsqueeze(0)
            anatomy = 2*anatomy.cuda() - 1.0

            predicted_img, predicted_label, _ = flow_model(base_anatomy, base_seg, anatomy)

            predicted_label = predicted_label.cpu().numpy()
            predicted_label = predicted_label[0, 0, ...]

            predicted_img = predicted_img.cpu().numpy()
            predicted_img = 0.5*predicted_img[0, 0, ...] + 0.5

            np.save(os.path.join(outdir, 'volume', 'pred_' + all_anatomies[i]), predicted_img)
            np.save(os.path.join(outdir, 'volume', 'pred_' + all_seg[i]), predicted_label)

            # visuals
            predicted_img_visuals = 255.0*generate_img_visuals(predicted_img)
            predicted_img_visuals = predicted_img_visuals.astype(np.uint8)

            predicted_seg_visuals = generate_seg_visuals(predicted_label)
            predicted_seg_visuals = predicted_seg_visuals.astype(np.uint8)

            Image.fromarray(predicted_img_visuals).save(os.path.join(outdir, 'visuals', 'pred_' + all_anatomies[i]+'.png'))
            Image.fromarray(predicted_seg_visuals).save(os.path.join(outdir, 'visuals', 'pred_' + all_seg[i]+'.png'))
            


analyse_autoencoder_results('../CANDIShare_clean/', '../BaselineAutoencoderResults/')