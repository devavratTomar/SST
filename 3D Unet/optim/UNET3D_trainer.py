from .base_trainer import BaseTrainer
from models import UNET3D

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from .utils import compute_dice_coeff, segm_to_image, DiceLoss3D, CrossEntropyLossWeighted
from kornia.losses import dice_loss


class UNET3D_trainer(BaseTrainer):

    def __init__(self, data_loader, logger, args, num_classes=17):
        super().__init__(data_loader, logger, args)

        self.num_classes = num_classes
        self.model = UNET3D(1, self.num_classes, downsample=False, norm="batch").to(self.device)

        self.criterion_ce = CrossEntropyLossWeighted()
        self.criterion_dice = DiceLoss3D()

        if args.optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(args.beta1,args.beta2), weight_decay=args.wd)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=args.patience, min_lr=args.min_lr, factor=0.5)

        self.input_ = torch.zeros(self.batch_size)
        self.segms = torch.zeros(self.batch_size)
        self.output = torch.zeros(self.batch_size)

        self.val_input_ = torch.zeros(self.batch_size)
        self.val_output = torch.zeros(self.batch_size)

        self.loss = torch.Tensor()
        self.train_loss = 0
        self.val_loss = 0

    def train_one_epoch(self):
        self.model.train()
        self.train_loss = 0
        for images, segms in self.tr:
            self._iter += 1
            self.input_ = images.to(self.device)

            self.segms = segms.to(self.device)
            self.output = self.model(self.input_)

            self.optimizer.zero_grad()

            self.loss = self.criterion_ce(self.output, self.segms) + 0.2*self.criterion_dice(self.output, self.segms)
            self.train_loss += images.shape[0] * self.loss.item()
            self.loss.backward()

            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            self.get_feedback()

        self.train_loss /= len(self.tr.dataset)

    def log(self):
        str_ = "Epoch {:d} | Step {:d} | Loss {:.5f}".format(self.cur_epoch, self._iter, self.loss.item())
        self.logger.info(str_)

    def visualize(self):
        self.vis.plot("Loss", "train", "unet loss", torch.Tensor([self._iter]), torch.Tensor([self.loss.item()]))
        if self._iter % 20 == 0:
            input_vis = ((self.input_.detach().cpu()[:,:,[8, 24, 30, 40, 56]]+1.0)/2.0).squeeze().unsqueeze(1)
            output_vis = segm_to_image(self.output.detach().cpu()[:,:,[8, 24, 30, 40, 56]], self.num_classes, _3D=True)
            gt_vis = segm_to_image(self.segms.detach().cpu()[:,[8, 24, 30, 40, 56]], self.num_classes, pred=False, _3D=True)
            self.vis.show_images("Train_input", input_vis)
            self.vis.show_images("Train_output", output_vis)
            self.vis.show_images("Train_GD", gt_vis)

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
        inters = torch.zeros(self.num_classes)
        tots = torch.zeros(self.num_classes)

        with torch.no_grad():
            for i, (images, segms) in enumerate(tqdm(self.val)):
                self.val_input_ = images.to(self.device)
                segms = segms.to(self.device)

                self.val_output = self.model(self.val_input_)

                self.val_loss += self.criterion_ce(self.val_output, segms).item() * images.shape[0] + 0.2*self.criterion_dice(self.val_output, segms).item()
                self.val_output = F.softmax(self.val_output, dim=1)

                new_inter, new_tot, new_gt_tots = compute_dice_coeff(self.val_output.cpu(), segms.cpu(), self.num_classes)
                inters += new_inter
                tots += new_tot

                if i == len(self.val) // 2:
                    input_vis = ((self.val_input_.cpu()[:,:,[8, 24, 30, 40, 56]]+1.0)/2.0).reshape(-1, self.val_input_.shape[1],
                                                                                                   self.val_input_.shape[-2],
                                                                                                   self.val_input_.shape[-1])
                    output_vis = segm_to_image(self.val_output.cpu()[:,:,[8, 24, 30, 40, 56]], self.num_classes, _3D=True)
                    gt_vis = segm_to_image(segms.cpu()[:,[8, 24, 30, 40, 56]], self.num_classes, pred=False, _3D=True)

        self.val_loss /= len(self.val.dataset)
        score = (2 * inters / tots) * 100
        self.logger.info(score)
        score = torch.mean(score).item()
        self.logger.info("Epoch {:d} | Dice score {:.5f}%".format(self.cur_epoch, score))

        if self.vis is not None:
            self.vis.plot("Loss", "train", "Epoch unet loss", torch.Tensor([self.cur_epoch]), torch.Tensor([self.train_loss]))
            self.vis.plot("Loss", "val", "Epoch unet loss", torch.Tensor([self.cur_epoch]), torch.Tensor([self.val_loss]))
            self.vis.plot("Dice score", "val", "Epoch unet dice score", torch.Tensor([self.cur_epoch]), torch.Tensor([score]))

            self.vis.show_images("Segmentation", output_vis[:8])
            self.vis.show_images("Real", input_vis[:8])
            self.vis.show_images("GD", gt_vis[:8])

        self.scheduler.step(self.val_loss)
