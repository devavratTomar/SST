import argparse
import torch
from models import UNET
from optim import load_trainer, compute_dice_coeff, compute_dice_score
from data import data_loader
from tqdm import tqdm

import torch.nn.functional as F
import numpy as np
import os

parser = argparse.ArgumentParser()

"""
ARGUMENTS FOR TRAINING
"""

"""
DATASET ARGUMENTS
"""
parser.add_argument("--root", "-r", default="../CANDIShare_clipped", help="The root for the dataset")
parser.add_argument("--dataset", default="2D_slice", choices=("2D_slice","3D", "P2D"), help="The name of the dataset")
parser.add_argument("--seed", default=0, type=int, help="Set the seed for training")
parser.add_argument("--load_all_in_memory", action="store_true", help="Whether or not we load all data in memory before training")
parser.add_argument("--pct_data", default=1.0, type=float, help="Percentage of data taken into account for generated data")
parser.add_argument("--pct_train", default=0.9, type=float, help="Percentage of data in train set")
parser.add_argument("--save_segments", action="store_true", help="Percentage of data in train set")

"""
BASIC SETTINGS
"""
parser.add_argument("--batch_size", default=16, type=int, help="The size of the batch to feed to the network")
parser.add_argument("--device", default="cuda", help="The device to train the model on")

"""
MODEL PARAMETERS
"""
parser.add_argument("--checkpoint", default=None, help="Specify a checkpoint if you want to start training on it")
parser.add_argument("--model", default="UNET", choices=("UNET", "UNET3D"))


args = parser.parse_args()

classes = ["Background", "Cerebral-WM","Cerebral-CX","Lateral-Vent","Cerebellum-WM","Cerebellum-CX","Thalamus-Proper",
           "Hippocampus","Amygdala","Caudate","Putamen","Pallidum","3rd-Vent","4th-Vent","Brain-Stem","CSF","VentralDC"]

test_loader = data_loader(args.root, args.dataset, args).test
num_classes = test_loader.dataset.num_classes
print(args.device)
model = UNET(1, num_classes).to(args.device)

checkpoint = torch.load(args.checkpoint, map_location=torch.device(args.device))
model.load_state_dict(checkpoint["unet"])
model.eval()

inters = torch.zeros(num_classes, len(test_loader.dataset.images))
tots = torch.zeros(num_classes, len(test_loader.dataset.images))
gt_tots = torch.zeros(num_classes, len(test_loader.dataset.images))

counter = 0
pred_vols = []
for x in range(len(test_loader.dataset.images)):
    pred_vols.append(torch.Tensor())

save_dir = args.checkpoint.split("/")[-3]

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

with torch.no_grad():
    for test, segms, index in tqdm(test_loader):
        test = test.to(args.device)

        output = F.softmax(model(test), dim=1)

        for i, img in enumerate(output):
            new_inters, new_tots, new_gt_tots = compute_dice_coeff(img.unsqueeze(0), segms[i].unsqueeze(0), num_classes)
            inters[:,index[i]] += new_inters
            tots[:,index[i]] += new_tots
            gt_tots[:,index[i]] += new_gt_tots

            pred_vols[index[i]] = torch.cat((pred_vols[index[i]], img.argmax(0).unsqueeze(0)),dim=0)

            if pred_vols[index[i]].shape[0]==test_loader.dataset.shape[0]:
                np.save(os.path.join(save_dir,test_loader.dataset.images[index[i]][:-7]+"_preds.npy"),pred_vols[index[i]].numpy())

tots[gt_tots==0]=0
scores_per_brain = compute_dice_score(inters[1:].numpy(), tots[1:].numpy())

np.save(os.path.join(save_dir,save_dir+"_results.npy"),scores_per_brain)

#print("Mean score across classes -> {:.3f}".format(torch.mean(scores).item()))
print("Mean score across brains -> {:.3f} | Std score across brains -> {:.3f}".format(np.nanmean(scores_per_brain),
                                                                                      np.std(np.nanmean(scores_per_brain, axis=0))))

for mean, std, class_ in zip(np.nanmean(scores_per_brain,axis=1).tolist(),
                             np.nanstd(scores_per_brain,axis=1).tolist(),classes[1:]):
    print("{:s} -> Mean : {:.3f} | Std : {:.3f} ".format(class_, mean, std))
