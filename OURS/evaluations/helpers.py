import random
import torch
from PIL import Image
from .constants import Constants
import numpy as np
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation
from medpy.metric.binary import dc


import os

from utilities.util import load_numpy

import skimage.segmentation as sks


K = Constants()

def convert_to_image(tensor):
    img = tensor[0][0][64]
    img = img.cpu().numpy()
    img = 0.5*img + 0.5
    img = 255.0*img
    img = img.astype(np.uint8)
    return Image.fromarray(img)

def color_seg(tensor):
    img = tensor[0][0][64]
    img = img.cpu().numpy()
    img = np.rint(img).astype(np.int)

    seg = K.colors[img].astype(np.uint8)
    return Image.fromarray(seg)

@torch.no_grad()
def random_color_jitter(img, brightness=[0.7, 1.2], contrast=[0.7, 1.2], saturation=[0.7, 1.1]):
    img = img.repeat(3, 1, 1, 1)

    brightness = random.uniform(brightness[0], brightness[1])
    contrast   = random.uniform(contrast[0], contrast[1])
    saturation = random.uniform(saturation[0], saturation[1])


    img = adjust_brightness(img, brightness)
    img = adjust_contrast(img, contrast)
    img = adjust_saturation(img, saturation)
    return img[0]

def compute_accuracy_brain(prediction, gt):
    # class-wise accuracies
    accuracies_overall = []

    accuracies_classwise = {
    }
    
    for [label, label_index] in K.brain_label_map:
        # leave background
        if label_index == 0:
            continue

        gt_mask = (gt == label_index)
        predict_mask = (prediction == label_index)

        if np.sum(gt_mask) == 0:
            accuracies_classwise[label] = []
            continue
        
        assert np.sum(gt_mask) != 0
        assert np.sum(predict_mask) !=0

        
        dice = dc(predict_mask, gt_mask)
        accuracies_overall.append(dice)

        if label not in accuracies_classwise:
            accuracies_classwise[label] = []
        
        accuracies_classwise[label].append(dice)

    return accuracies_overall, accuracies_classwise

def overlay_img_seg(img, seg):
    colored_seg = np.zeros_like(seg)
    for i in np.unique(seg):
        # get seg mask for this label
        if i == 0:
            continue
        mask = seg == i
        mask_boundary = sks.find_boundaries(mask, mode='inner').astype(int)
        colored_seg = colored_seg + i*mask_boundary
    
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    
    img = K.colors[colored_seg.astype(int)]*(colored_seg[:,:, None] != 0) + img*(colored_seg[:,:, None] == 0)
    return img.astype(np.uint8)


def concat_all(results):
    n_imgs = len(results['gt'])
    out = []
    def get_all_items_at_ixd(idx):
        items = []
        for k in results.keys():
            items.append(results[k][idx])
        
        return items
    
    for i in range(n_imgs):
        img_results = get_all_items_at_ixd(i)
        concat_imgs = np.concatenate(img_results, axis=1)
        concat_imgs = concat_imgs.astype(np.uint8)
        out.append(concat_imgs)
    
    return out

@torch.no_grad()
def evaluate_seg(img_dir, output_dir, results_dirs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    test_imgs = sorted([os.path.join(img_dir, i + '_MR.npy.gz') for i in K.brain_tests])
    test_segs = sorted([os.path.join(img_dir, i + '_MR_seg.npy.gz') for i in K.brain_tests])

    all_results = {}
    all_results['gt'] = []
    for rdir in results_dirs:
        all_results[os.path.basename(rdir)] = []

# fileter filenames
    file_names = [f.split('_')[0] for f in os.listdir(results_dirs[0]) if f.endswith(('_MR_seg_preds.npy', '_MR_pred.npy', '_MR_preds.npy', '_MR__preds.npy', 'MR_seg.npy.gz'))]
    
    test_imgs = sorted([i for i in test_imgs if any(f_name in i for f_name in file_names)])
    test_segs = sorted([i for i in test_segs if any(f_name in i for f_name in file_names)])

    print(test_imgs)
    for i in range(len(test_imgs)):
        test_img_sample = load_numpy(test_imgs[i]).transpose(2, 1, 0)[64]
        test_img_sample = 255.0*(test_img_sample - test_img_sample.min())/(test_img_sample.max() - test_img_sample.min())
        test_img_sample = test_img_sample.astype(np.uint8)
        test_seg_sample = np.rint(load_numpy(test_segs[i])).transpose(2, 1, 0)[64]
        gt = overlay_img_seg(test_img_sample, test_seg_sample)
        all_results['gt'].append(gt)

    
    for results_dir in results_dirs:
        results_seg = sorted([os.path.join(results_dir, i) for i in os.listdir(results_dir) if any(f_name in i for f_name in file_names) and i.endswith(('_MR_seg_preds.npy', '_MR_pred.npy', '_MR_preds.npy', '_MR__preds.npy', 'MR_seg.npy.gz'))])
        for i in range(len(results_seg)):
            test_img_sample = load_numpy(test_imgs[i]).transpose(2, 1, 0)[64]
            test_img_sample = 255.0*(test_img_sample - test_img_sample.min())/(test_img_sample.max() - test_img_sample.min())
            
            segs = np.load(results_seg[i]) if results_seg[i].endswith('npy') else load_numpy(results_seg[i])
            print(segs.shape)
            if segs.shape[0] == 160:
                segs = segs.transpose(2, 1, 0)
            
            result_seg_sample = np.rint(segs[64])
            print(results_dir, np.unique(result_seg_sample))
            pred = overlay_img_seg(test_img_sample, result_seg_sample)
            all_results[os.path.basename(results_dir)].append(pred)

    out = concat_all(all_results)
    for i, img in enumerate(out):
        print(img.shape)
        Image.fromarray(img).save(os.path.join(output_dir, str(i)+'.png'))
