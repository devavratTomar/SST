import argparse
import torch
import glob
from models import UNET, UNET3D
from optim import load_trainer, compute_dice_coeff, compute_dice_score
from data import data_loader
from tqdm import tqdm
from oasis_data import *

import torch.nn.functional as F
import numpy as np
import os
import gzip

def load_numpy(path):
    f = gzip.GzipFile(path, "r", compresslevel=1)
    tmp = np.load(f).astype(np.float32)
    f.close()
    return tmp

parser = argparse.ArgumentParser()

"""
ARGUMENTS FOR TRAINING
"""

"""
DATASET ARGUMENTS
"""
parser.add_argument("--dataset_name", default="CANDI", choices=("OASIS","OASIS_generated", "CANDI","CANDI_generated"), help="The name of the dataset")

"""
BASIC SETTINGS
"""
parser.add_argument("--batch_size", default=32, type=int, help="The size of the batch to feed to the network")
parser.add_argument("--device", default="cuda", help="The device to train the model on")

"""
MODEL PARAMETERS
"""
parser.add_argument("--checkpoint", default=None, help="Specify a checkpoint if you want to start training on it")
parser.add_argument("--model", default="UNET3D", choices=("UNET", "UNET3D"))


args = parser.parse_args()

classes = ["Background", "Cerebral-WM","Cerebral-CX","Lateral-Vent","Cerebellum-WM","Cerebellum-CX","Thalamus-Proper",
           "Hippocampus","Amygdala","Caudate","Putamen","Pallidum","3rd-Vent","4th-Vent","Brain-Stem","CSF","VentralDC"]

if args.dataset_name == "OASIS":
    test_root = "../OASIS_clean"
    test_vols_files_dir = os.path.join(test_root, '*_MR.npy.gz')
    test_seg_files_dir = os.path.join(test_root, '*_MR_seg.npy.gz')
    test_images = np.array(sorted([x for x in glob.glob(test_vols_files_dir) if x.split("/")[-1].split('_')[0] in test_samples]))
    test_segments = np.array(sorted([x for x in glob.glob(test_seg_files_dir) if x.split("/")[-1].split('_')[0] in test_samples]))
elif args.dataset_name == "CANDI":
    test_root = "../CANDISHARE_clean/CANDIShare_clean_gz/test"

    test_vols_files_dir = os.path.join(test_root, '*_MR.npy.gz')
    test_seg_files_dir = os.path.join(test_root, '*_MR_seg.npy.gz')
    test_images = np.array(sorted([x for x in glob.glob(test_vols_files_dir)]))
    test_segments = np.array(sorted([x for x in glob.glob(test_seg_files_dir)]))

num_classes = len(classes)
if args.model == "UNET":
    model = UNET(1, num_classes).to(args.device)
else:
    model = UNET3D(1, num_classes, downsample=False, norm="batch").to(args.device)

checkpoint = torch.load(args.checkpoint, map_location=torch.device(args.device))
model.load_state_dict(checkpoint["unet"])
model.eval()

save_dir = os.path.join("Results",args.checkpoint.split("/")[-3])

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

inters = torch.zeros(num_classes, len(test_images))
tots = torch.zeros(num_classes, len(test_images))
gt_tots = torch.zeros(num_classes, len(test_images))

pred_vols = []
for x in range(len(test_images)):
    pred_vols.append(torch.Tensor())

with torch.no_grad():
    for i, scan in enumerate(tqdm(test_images)):
        
        img = load_numpy(scan).transpose(2,1,0)
        img = torch.from_numpy(img).to(torch.float32)

        segment = np.rint(load_numpy(test_segments[i])).transpose(2,1,0)
        segment = torch.from_numpy(segment).to(torch.long)

        img = 2.0 * img - 1.0

        if args.model == "UNET":
            for j, index in enumerate(range(0,len(img),args.batch_size)):
                start_idx = j*args.batch_size
                end_idx=min(len(img),(j+1)*args.batch_size)
                input_ = img[start_idx:end_idx].to(args.device).unsqueeze(1)
                seg_ = segment[start_idx:end_idx].to(args.device)

                output = F.softmax(model(input_),dim=1)
                pred_vols[i] = torch.cat((pred_vols[i].cpu(), output.argmax(0).unsqueeze(0).cpu()),dim=0)

                new_inters, new_tots, new_gt_tots = compute_dice_coeff(output.cpu(), seg_.cpu(), num_classes)
                inters[:,i] += new_inters
                tots[:,i] += new_tots
                gt_tots[:,i] += new_gt_tots
        else:
            input_ = img.unsqueeze(0).unsqueeze(0)
            seg_ = segment.unsqueeze(0)
            output = F.softmax(model(input_.to(args.device)), dim=1)

            pred_vols[i] = output[0].argmax(0).cpu()
            new_inters, new_tots, new_gt_tots = compute_dice_coeff(output.cpu(), seg_.cpu(), num_classes)
            inters[:,i] = new_inters
            tots[:,i] = new_tots
            gt_tots[:,i] = new_gt_tots

        np.save(os.path.join(save_dir,scan.split("/")[-1][:-7]+"_preds.npy"),pred_vols[i].numpy())

tots[gt_tots==0]=0
scores_per_brain = compute_dice_score(inters[1:].numpy(), tots[1:].numpy())

np.save(os.path.join(save_dir+"_results.npy"),scores_per_brain)

print("Mean score across brains -> {:.3f} | Std score across brains -> {:.3f}".format(np.nanmean(scores_per_brain),
                                                                                      np.std(np.nanmean(scores_per_brain, axis=0))))

for mean, std, class_ in zip(np.nanmean(scores_per_brain,axis=1).tolist(),
                             np.nanstd(scores_per_brain,axis=1).tolist(),classes[1:]):
    print("{:s} -> Mean : {:.3f} | Std : {:.3f} ".format(class_, mean, std))
