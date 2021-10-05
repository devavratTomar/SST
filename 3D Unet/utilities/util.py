import os
import torch
from pathlib import Path
import numpy as np
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]

def natural_sort(items):
    items.sort(key=natural_keys)

def load_network(net, label, epoch, checkpoints_dir):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(checkpoints_dir, 'models')
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=True):
    def tile_images(imgs, picturesPerRow=4):
        if imgs.shape[0] % picturesPerRow == 0:
            rowPadding = 0
        else:
            rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
        if rowPadding > 0:
            imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

        # Tiling Loop (The conditionals are not necessary anymore)
        tiled = []
        for i in range(0, imgs.shape[0], picturesPerRow):
            tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

        tiled = np.concatenate(tiled, axis=0)
        return tiled

    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np
        
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) - image_numpy.min()) / (image_numpy.max() - image_numpy.min() + 1e-6) * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)
