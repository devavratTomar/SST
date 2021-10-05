import argparse
import logging
import os, sys

from data import data_loader
from optim import load_trainer

parser = argparse.ArgumentParser()

"""
ARGUMENTS FOR TRAINING
"""

"""
DATASET ARGUMENTS
"""
parser.add_argument("--root", "-r", default="../OASIS_clean", help="The root for the dataset")
parser.add_argument("--dataset", default="3D", choices=("2D_slice","3D","P2D","2D_slice_generated"), help="The name of the dataset")
parser.add_argument("--dataset_name", default="OASIS_generated", choices=("OASIS","OASIS_generated", "CANDI","CANDI_generated"), help="The name of the dataset")
parser.add_argument("--seed", default=0, type=int, help="Set the seed for training")
parser.add_argument("--normalization", default="normal", choices=("normal", "resnet"), help="Specify the image normalization")
parser.add_argument("--load_all_in_memory", action="store_true", help="Whether or not we load all data in memory before training")
parser.add_argument("--pct_data", default=1.0, type=float, help="Percentage of data taken into account for generated data")
parser.add_argument("--pct_train", default=1.0, type=float, help="Percentage of data in train set")

"""
BASIC SETTINGS
"""
parser.add_argument("--batch_size", default=1, type=int, help="The size of the batch to feed to the network")
parser.add_argument("--device", default="cuda", help="The device to train the model on")
parser.add_argument("--epochs", default=100, type=int, help="The number of epochs to train")
parser.add_argument("--lr", default=0.0002, type=float, help="the learning rate for training")
parser.add_argument("--beta1", default=0.9, type=float, help="beta1 argument for adam optimizer")
parser.add_argument("--beta2", default=0.999, type=float, help="beta2 argument for adam optimizer")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--min_lr", default=0.0001, type=float, help="The minimum learning rate for training")
parser.add_argument("--patience", default=3, type=float, help="The patience for the scheduler to decrease the learning rate")
parser.add_argument("--weighted", action="store_true", help="Whether or not we use class balance in the training")
parser.add_argument("--optimizer", default="Adam", choices=("SGD", "Adam"), help="The optimizer for training")

"""
MODEL PARAMETERS
"""
parser.add_argument("--checkpoint", default=None, help="Specify a checkpoint if you want to start training on it")
parser.add_argument("--check_interval", default=1, type=int, help="Specify the checkpoint saving rate (with respect to the epoch)")
parser.add_argument("--model", default="UNET3D", choices=("UNET", "UNET3D"))
parser.add_argument("--inch", default=1, type=int, help="The number of channels in input")
"""
FEEDBACK SETTINGS
"""
parser.add_argument("--exp", default="OASIS_UNET_NEW_3D_1X", help="Specify the path to save checkpoint and logs for the "
                                                         "experiment")
parser.add_argument("--visualize", action="store_true", help="Whether we visualize the training plots or not")
parser.add_argument("--port", default=1074, type=int, help="Specify the port on which you want to connect to visdom server")

args = parser.parse_args()

args.visualize = True

"""
Arguments preprocessing
"""
args.exp_dir = os.path.join("Experiments", args.exp)
if not os.path.isdir(args.exp_dir):
    os.mkdir(args.exp_dir)
folders = os.listdir(args.exp_dir)
if "Images" not in folders:
    os.mkdir(os.path.join(args.exp_dir, "Images"))
if "Checkpoints" not in folders:
    os.mkdir(os.path.join(args.exp_dir, "Checkpoints"))
if "VisdomLogs" not in folders:
    os.mkdir(os.path.join(args.exp_dir, "VisdomLogs"))

logging.basicConfig(filename=os.path.join(args.exp_dir, args.exp+"_.log"),
                    level=logging.INFO,
                    format='%(asctime)s %(msecs)d- %(process)d-%(levelname)s - %(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info("Arguments parsed")
logger.info("{:s} folder ready".format(args.exp))

"""
LOAD dataset
"""
logger.info("Load datasets")
loaders = data_loader(args.root, args.dataset, args)
logger.info("Datasets loaded")


logger.info("Load trainer")
trainer = load_trainer(args.model, loaders, logger, args)
logger.info("Trainer loaded")

trainer.train()
