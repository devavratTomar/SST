from torch.utils.data import Dataset
from utilities.util import natural_sort
import os
from .transformations import RandomSmoothFlowPair3D, RandomColorJitterPair, RandomColorJitter
import numpy as np
import torch
import skimage.segmentation as sks
import random
import torchvision as tv

from evaluations.constants import Constants

from utilities.util import load_numpy

class MRIDataset(Dataset):
    """
    Class to load 3d numpy columes of MRIDataset
    """
    def __init__(self, root_dir, size, base_files, dataset_name, train=True):
        self.anatomies = [f for f in os.listdir(root_dir) if f.endswith('MR.npy.gz')]
        self.segs      = [f for f in os.listdir(root_dir) if f.endswith('MR_seg.npy.gz')]
        
        K = Constants(dataset_name)
        # remove the test cases
        for test in K.brain_tests:
            if test in self.anatomies:
                self.anatomies.remove(test + '_MR.npy.gz')
                self.segs.remove(test + '_MR_seg.npy.gz')

        self.rootdir = root_dir

        # sort to make sure the names align
        natural_sort(self.anatomies)
        natural_sort(self.segs)

        # start poping from the end to avoid the need of reindexing
        indices_base_anatomy = sorted([self.anatomies.index(f + '.npy.gz') for f in base_files], reverse=True)
        indices_base_seg     = sorted([self.segs.index(f + '_seg.npy.gz') for f in base_files], reverse=True)

        self.base_anatomies = [self.anatomies.pop(i) for i in indices_base_anatomy]
        self.base_segs      = [self.segs.pop(i) for i in indices_base_seg]

        self.random_transform = RandomSmoothFlowPair3D(size=size, flow_sigma=1, blur_sigma=5)

    def __getitem__(self, index):
        index_base = index % len(self.base_segs)
        
        # load one of the base anatomy
        base_anatomy = load_numpy(os.path.join(self.rootdir, self.base_anatomies[index_base])).transpose(2, 1, 0)
        base_seg = load_numpy(os.path.join(self.rootdir, self.base_segs[index_base])).transpose(2, 1, 0)

        # add channel axis
        base_anatomy = torch.from_numpy(base_anatomy).to(torch.float32).unsqueeze(0)
        base_seg = torch.from_numpy(base_seg).to(torch.float32).unsqueeze(0)

        # load real data and perfrom random flow data augmentation
        real_anatomy = torch.from_numpy(load_numpy(os.path.join(self.rootdir,self.anatomies[index])).transpose(2, 1, 0)).to(torch.float32)
        
        real_anatomy = self.random_transform(real_anatomy)
        
        real_anatomy = 2.0*real_anatomy - 1.0
        base_anatomy = 2.0*base_anatomy - 1.0

        return base_anatomy, base_seg, real_anatomy

    def __len__(self):
        return len(self.anatomies)


class MRIDatasetRandom(Dataset):
    """
    Loads images with random moving and fixed images.
    """
    def __init__(self, root_dir, size, dataset_name, train=True):
        self.anatomies = [f for f in os.listdir(root_dir) if f.endswith('MR.npy.gz')]
        self.rootdir = root_dir
        K = Constants(dataset_name)
        # remove the test cases
        for test in K.brain_tests:
            if test in self.anatomies:
                self.anatomies.remove(test + '_MR.npy.gz')

        self.anatomies = [load_numpy(os.path.join(root_dir, f)).transpose(2, 1, 0) for f in self.anatomies]

        paired_anatomies = []

        for i in range(len(self.anatomies)):
            for j in range(len(self.anatomies)):
                paired_anatomies.append([self.anatomies[i], self.anatomies[j]])
        
        self.anatomies = paired_anatomies

        self.len = len(self.anatomies)

        self.random_color_jitter_pair = RandomColorJitterPair()


    def __getitem__(self, index):
        # moving_path, fixed_path = self.anatomies[index]

        # # load one of the base anatomy
        # moving_anatomy = load_numpy(os.path.join(self.rootdir, moving_path)).transpose(2, 1, 0)
        # fixed_anatomy = load_numpy(os.path.join(self.rootdir, fixed_path)).transpose(2, 1, 0)
        moving_anatomy, fixed_anatomy = self.anatomies[index]
        # pytorch tensors
        moving_anatomy = torch.from_numpy(moving_anatomy).to(torch.float32)
        fixed_anatomy = torch.from_numpy(fixed_anatomy).to(torch.float32)

        moving_anatomy, fixed_anatomy = self.random_color_jitter_pair([moving_anatomy, fixed_anatomy])

        moving_anatomy = moving_anatomy.unsqueeze(0)
        fixed_anatomy= fixed_anatomy.unsqueeze(0)
        
        moving_anatomy = 2.0*moving_anatomy - 1.0
        fixed_anatomy = 2.0*fixed_anatomy - 1.0

        return moving_anatomy, torch.zeros_like(moving_anatomy), fixed_anatomy

    def __len__(self):
        return self.len

class MRIDatasetClass(Dataset):
    """[This is used for training style encoder. We do offline transformation to generate positive pairs having same styles.]

    Args:
        Dataset ([type]): [root directory of *_StyleAug]
    """
    def __init__(self, rootdir):
        classes = sorted([c for c in os.listdir(rootdir) if c.endswith('MR')])
        
        self.class_to_imgs = {}
        self.all_imgs = []

        for c in classes:
            imgs_class = sorted([os.path.join(rootdir, c, f) for f in os.listdir(os.path.join(rootdir, c)) if f.endswith('atlas.npy.gz')])
            self.class_to_imgs[c] = imgs_class
            self.all_imgs += imgs_class

    def __getitem__(self, index):
        # find the class of the current index
        img_path = self.all_imgs[index]
        
        # randomly select another img from this class
        class_name = os.path.basename(os.path.dirname(img_path))
        key_img_path = random.choice(self.class_to_imgs[class_name])

        q = load_numpy(img_path)
        k = load_numpy(key_img_path)       

        # pytorch tensors
        q = torch.from_numpy(q).to(torch.float32)
        k = torch.from_numpy(k).to(torch.float32)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)

        return q, k
        

    def __len__(self):
        return len(self.all_imgs)


class MRIDatasetClassBaseAtlas(Dataset):
    """[Same as MRIDatasetClass, but also outputs base image]

    Args:
        Dataset ([type]): [*_StyleAug]
    """
    def __init__(self, rootdir, base_files):
        self.base_files = base_files

        classes = sorted([c for c in os.listdir(rootdir) if c.endswith('MR')])
        self.class_to_imgs = {}
        self.all_imgs = []
        self.all_segs = []

        for c in classes:
            imgs_class = sorted([os.path.join(rootdir, c, f) for f in os.listdir(os.path.join(rootdir, c)) if f.endswith('atlas.npy.gz')])
            imgs_class_seg = sorted([os.path.join(rootdir, c, f) for f in os.listdir(os.path.join(rootdir, c)) if f.endswith('seg.npy.gz')])

            self.class_to_imgs[c] = [imgs_class, imgs_class_seg]
            
            if c not in base_files:
                self.all_imgs += imgs_class
                self.all_segs += imgs_class_seg
        
        self.base_imgs = {}
        
        # take out the base atlases
        for f in base_files:
            self.base_imgs[f] = [os.path.join(rootdir, f, 'base_atlas.npy.gz'), os.path.join(rootdir, f, 'base_seg.npy.gz')]
        
        print(self.base_imgs)
        
    def __getitem__(self, index):
        # find the class of the current index
        img_path = self.all_imgs[index]
        
        # randomly select another img from this class
        class_name = os.path.basename(os.path.dirname(img_path))
        key_img_path = random.choice(self.class_to_imgs[class_name][0])

        q = load_numpy(img_path)
        k = load_numpy(key_img_path)
        
        # pytorch tensors
        q = torch.from_numpy(q).to(torch.float32)
        k = torch.from_numpy(k).to(torch.float32)

        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        
        # base image and base atlas
        base_index = random.randint(0, len(self.base_files)-1)
        
        base_img = load_numpy(self.base_imgs[self.base_files[base_index]][0])
        base_seg = load_numpy(self.base_imgs[self.base_files[base_index]][1])

        base_img = torch.from_numpy(base_img).to(torch.float32).unsqueeze(0)
        base_seg = torch.from_numpy(base_seg).to(torch.float32).unsqueeze(0)

        return base_img, base_seg, q, k
        

    def __len__(self):
        return len(self.all_imgs)

class MRIDatasetRandomBaseAtlas(Dataset):
    """
    Loads images with random moving and fixed images along with base atlas.
    """

    def __init__(self, root_dir, base_names, base_path, dataset_name, train=True):
        self.anatomies = [f for f in os.listdir(root_dir) if f.endswith('MR.npy.gz')]
        self.rootdir = root_dir

        if base_path == '':
            base_path = root_dir
        
        K = Constants(dataset_name)
        # remove the test cases
        for test in K.brain_tests:
            if test in self.anatomies:
                self.anatomies.remove(test + '_MR.npy.gz')
        
        # base atlas and segmentation
        self.base_imgs = [[load_numpy(os.path.join(base_path, f + '.npy.gz')).transpose(2, 1, 0),\
                           load_numpy(os.path.join(base_path, f + '_seg.npy.gz')).transpose(2, 1, 0)]\
                           for f in base_names]
        
        paired_anatomies = []

        for i in range(len(self.anatomies)):
            for j in range(len(self.anatomies)):
                paired_anatomies.append([self.anatomies[i], self.anatomies[j]])
        
        self.anatomies = paired_anatomies

        self.len = len(self.anatomies)

        self.random_color_jitter = RandomColorJitter()

    def __getitem__(self, index):
        moving_path, fixed_path = self.anatomies[index]

        # load one of the base anatomy
        moving_anatomy = load_numpy(os.path.join(self.rootdir, moving_path)).transpose(2, 1, 0)
        fixed_anatomy = load_numpy(os.path.join(self.rootdir, fixed_path)).transpose(2, 1, 0)

        # pytorch tensors
        moving_anatomy = torch.from_numpy(moving_anatomy).to(torch.float32)
        fixed_anatomy = torch.from_numpy(fixed_anatomy).to(torch.float32)

        moving_anatomy, fixed_anatomy = self.random_color_jitter(moving_anatomy), self.random_color_jitter(fixed_anatomy)

        moving_anatomy = moving_anatomy.unsqueeze(0)
        fixed_anatomy= fixed_anatomy.unsqueeze(0)
        
        moving_anatomy = 2.0*moving_anatomy - 1.0
        fixed_anatomy = 2.0*fixed_anatomy - 1.0

        base_index = random.randint(0, len(self.base_imgs)-1)
        base_img = 2.0*torch.from_numpy(self.base_imgs[base_index][0]).to(torch.float32).unsqueeze(0) - 1.0
        base_seg = torch.from_numpy(self.base_imgs[base_index][1]).to(torch.float32).unsqueeze(0)

        return base_img, base_seg, moving_anatomy, fixed_anatomy

    def __len__(self):
        return len(self.anatomies)