import abc
import torch
import torch.nn as nn
import os

from visualization import VisdomVisualizer


class BaseTrainer(abc.ABC):

    def __init__(self, data_loader, logger, args):

        self.tr = data_loader.train

        self.val = data_loader.val

        self.epochs = args.epochs
        self.cur_epoch = 0
        self._iter = 0
        self.batch_size = args.batch_size
        self.device = args.device

        self.checkpoint_save = os.path.join(args.exp_dir, "Checkpoints")
        self.logger = logger
        self.vis = None
        if args.visualize:
            self.vis = VisdomVisualizer(args.exp_dir, args.port)
            self.vis_json_file = os.path.join(args.exp_dir, "VisdomLogs", args.exp + "_vis_")

        self.checkpoint = args.checkpoint
        self.checkpoint_exist = False
        if self.checkpoint is not None and os.path.exists(self.checkpoint):
            self.checkpoint = torch.load(self.checkpoint, map_location=torch.device(args.device))
            self.checkpoint_exist = True

        self.check_interval = args.check_interval

        #self.mean = torch.tensor(self.source_tr.dataset.mean).reshape(1, -1, 1, 1)
        #self.std = torch.tensor(self.source_tr.dataset.std).reshape(1, -1, 1, 1)
        #self.size = self.source_tr.dataset.size

        self.args = args

    def train(self):
        self.load_model()
        self.logger.info("Loaded checkpoint")
        self.logger.info("Start training")
        starter = self.cur_epoch+1
        for e in range(starter, self.epochs + 1):
            self.cur_epoch = e
            self.train_one_epoch()
            if self.check_interval > 0 and e % self.check_interval == 0:
                self.save_model()
            # We save the visualization json file
            if self.args.visualize:
                self.vis.save_vis(self.vis_json_file + str(self.cur_epoch) + ".json")
            self.validate()

    def get_feedback(self):
        self.log()
        if self.vis is not None:
            self.visualize()

    @abc.abstractmethod
    def train_one_epoch(self):
        pass

    @abc.abstractmethod
    def save_model(self):
        pass

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def visualize(self):
        pass

    @abc.abstractmethod
    def log(self):
        pass

    @abc.abstractmethod
    def validate(self):
        pass

    def save_checkpoint(self, dic, filename):
        torch.save(dic, os.path.join(self.checkpoint_save, filename))

    def load_checkpoint(self, key, model=None):
        if self.checkpoint_exist:
            if isinstance(model, nn.Module):
                return model.load_state_dict(self.checkpoint[key])
            else:
                return self.checkpoint.get(key, 0)
        else:
            return 0
