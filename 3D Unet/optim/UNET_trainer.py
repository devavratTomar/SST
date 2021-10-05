from .base_trainer import BaseTrainer
from models import UNET

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from .utils import compute_dice_coeff, segm_to_image, compute_dice_score
import numpy as np


class UNET2D_trainer(BaseTrainer):

    def __init__(self, data_loader, logger, args):
        super().__init__(data_loader, logger, args)

        self.num_classes = self.tr.dataset.num_classes
        self.model = UNET(1, self.num_classes, downsample=False, norm="inst").to(self.device)

        torch.backends.cudnn.benchmark = True

        if args.weighted:
            self.criterion = nn.CrossEntropyLoss(weight=self.tr.dataset.weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        if args.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1,args.beta2), weight_decay=args.wd)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=args.patience, min_lr=args.min_lr, factor=0.5)

        self.input_ = torch.zeros(self.batch_size).to(self.device)
        self.segms = torch.zeros(self.batch_size).to(self.device)
        self.output = torch.zeros(self.batch_size).to(self.device)

        self.val_input_ = torch.zeros(self.batch_size)
        self.val_output = torch.zeros(self.batch_size)

        self.loss = torch.Tensor()
        self.train_loss = 0
        self.val_loss = 0

    def train_one_epoch(self):
        self.model.train()
        self.train_loss = 0
        for images, segms, _ in self.tr:
            self._iter += 1
            self.input_ = images.to(self.device)
            self.segms = segms.to(self.device)
            self.output = self.model(self.input_)

            self.optimizer.zero_grad()
            self.loss = self.criterion(self.output, self.segms)
            self.train_loss += images.shape[0] * self.loss.item()
            self.loss.backward()

            self.optimizer.step()

            self.get_feedback()

        self.train_loss /= len(self.tr.dataset)

    def log(self):
        str_ = "Epoch {:d} | Step {:d} | Loss {:.5f}".format(self.cur_epoch, self._iter, self.loss.item())
        self.logger.info(str_)

    def visualize(self):
        self.vis.plot("Loss", "train", "unet loss", torch.Tensor([self._iter]), torch.Tensor([self.loss.item()]))
        if self._iter % 20 == 0:
            self.vis.show_images("Tr_Real", (self.input_[:8]+1)/2.0)
            self.vis.show_images("Tr_Segmentation", segm_to_image(self.output[:8].detach().cpu(), self.num_classes))
            self.vis.show_images("Tr_GD", segm_to_image(self.segms[:8].detach().cpu(), self.num_classes, pred=False))

    def save_model(self):
        dic = {"unet": self.model.state_dict(),
               "epoch": self.cur_epoch,
               "iter": self._iter}
        self.save_checkpoint(dic, "UNET_{:d}.pth".format(self.cur_epoch))

    def load_model(self):
        self.cur_epoch = self.load_checkpoint("epoch")
        self._iter = self.load_checkpoint("item")
        self.load_checkpoint("unet", self.model)

    def validate(self):
        self.model.eval()
        self.val_loss = 0
        inters = torch.zeros(self.num_classes, len(self.val.dataset.images))
        tots = torch.zeros(self.num_classes, len(self.val.dataset.images))
        gt_tots = torch.zeros(self.num_classes, len(self.val.dataset.images))

        with torch.no_grad():
            for i, (images, segms, index) in enumerate(tqdm(self.val)):
                self.val_input_ = images.to(self.device)
                segms = segms.to(self.device)

                self.val_output = self.model(self.val_input_)

                if self.args.weighted:
                    self.val_loss += F.cross_entropy(self.val_output,segms, weight=self.val.dataset.weights.to(self.device)).item() * images.shape[0]
                else:
                    self.val_loss += F.cross_entropy(self.val_output,segms).item() * images.shape[0]

                self.val_output = F.softmax(self.val_output, dim=1)

                for j, img in enumerate(self.val_output):
                    new_inters, new_tots, new_gt_tots = compute_dice_coeff(img.unsqueeze(0).cpu(), segms[j].unsqueeze(0).cpu(), self.num_classes)
                    inters[:,index[j]] += new_inters
                    tots[:,index[j]] += new_tots
                    gt_tots[:,index[j]] += new_gt_tots 

                if i == len(self.val) // 2:
                    input_vis = (self.val_input_.cpu()+1.0)/2.0
                    output_vis = segm_to_image(self.val_output.cpu(), self.num_classes)
                    gt_vis = segm_to_image(segms.cpu(), self.num_classes, pred=False)

        self.val_loss /= len(self.val.dataset)
        tots[gt_tots==0]=0
        scores_per_brain = compute_dice_score(inters[1:].numpy(), tots[1:].numpy())
        self.logger.info(np.nanmean(scores_per_brain,axis=1).tolist())
        score = np.nanmean(scores_per_brain)
        self.logger.info("Epoch {:d} | Dice score {:.5f}%".format(self.cur_epoch, score))

        if self.vis is not None:
            self.vis.plot("Loss", "train", "Epoch unet loss", torch.Tensor([self.cur_epoch]), torch.Tensor([self.train_loss]))
            self.vis.plot("Loss", "val", "Epoch unet loss", torch.Tensor([self.cur_epoch]), torch.Tensor([self.val_loss]))
            self.vis.plot("Dice score", "val", "Epoch unet dice score", torch.Tensor([self.cur_epoch]), torch.Tensor([score]))

            self.vis.show_images("Segmentation", output_vis[:8])
            self.vis.show_images("Real", input_vis[:8])
            self.vis.show_images("GD", gt_vis[:8])

        self.scheduler.step(self.val_loss)
        self.logger.info("LR : {:.8f}".format(self.optimizer.param_groups[0]['lr']))
