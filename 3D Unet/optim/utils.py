import torch
import numpy as np

from matplotlib import cm
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def segm_to_image(segments, num_classes, original=None, pred=True, _3D=False, add_back=False):
    if _3D:
        if pred:
            segments = segments.permute(0, 2, 1, 3, 4).reshape(-1, segments.shape[1], segments.shape[3], segments.shape[4])
        else:
            segments = segments.reshape(-1, segments.shape[2], segments.shape[3])
    if pred:
        segments = torch.argmax(segments, dim=1)
    new_segments = torch.zeros(segments.shape[0], segments.shape[1],segments.shape[2], 3)
    color_map = cm.get_cmap('gist_rainbow')

    for i in range(1, num_classes):
        color = torch.Tensor(color_map(1. * i / num_classes)[:-1])
        new_segments[segments == i] = color

    black = torch.Tensor([0, 0, 0])
    if not add_back:
        new_segments[segments == 0] = black
    elif add_back and original is not None:
        new_segments[original == 0] = black

    return new_segments.permute(0,3,1,2)


def compute_dice_coeff(output, gt, num_classes, original=None, predict_back=True):
    assert (num_classes == output.shape[1])
    inters = torch.zeros(num_classes)
    tots = torch.zeros(num_classes)
    gt_tot = torch.zeros(num_classes)

    output = torch.argmax(output, dim=1)
    if not predict_back and original is not None:
        output[original == 0] = 0

    for k in range(num_classes):
        inters[k] = torch.sum(output[gt == k] == k)
        tots[k] = torch.sum(gt[gt == k] == k) + torch.sum(output[output == k] == k)
        gt_tot[k] += torch.sum(gt[gt == k] == k)


    return inters, tots, gt_tot

def compute_dice_score(inters, tots):
    tots[tots==0] = np.nan
    return 200*inters / tots


def eval(test_loader, num_classes, model, args):
    inters = torch.zeros(num_classes, len(test_loader.dataset.images))
    tots = torch.zeros(num_classes, len(test_loader.dataset.images))
    gt_tots = torch.zeros(num_classes, len(test_loader.dataset.images))

    with torch.no_grad():
        for test, segms, index in tqdm(test_loader):
            test = test.to(args.device)

            output = F.softmax(model(test), dim=1)

            for i, img in enumerate(output):
                new_inters, new_tots, new_gt_tots = compute_dice_coeff(img.unsqueeze(0), segms[i].unsqueeze(0), num_classes)
                inters[:,index[i]] += new_inters
                tots[:,index[i]] += new_tots
                gt_tots[:,index[i]] += new_gt_tots


    tots[gt_tots==0]=0
    scores_per_brain = compute_dice_score(inters[1:].numpy(), tots[1:].numpy())

    return scores_per_brain


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.utils.one_hot import one_hot


# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

def dice_loss_3d(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Args:
        input (torch.Tensor): logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        labels (torch.Tensor): labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.

    Return:
        torch.Tensor: the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = dice_loss(input, target)
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxNxSxHxW. Got: {}"
                         .format(input.shape))

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {} and {}"
                         .format(input.shape, input.shape))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3, 4)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2. * intersection / (cardinality + eps)

    return torch.mean(-dice_score + 1.)



class DiceLoss3D(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Args:
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> criterion = DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super(DiceLoss3D, self).__init__()
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss_3d(input, target, self.eps)


class CrossEntropyLossWeighted(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self):
        super(CrossEntropyLossWeighted, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        target_one_hot: torch.Tensor = one_hot(targets, num_classes=inputs.shape[1],device=inputs.device, dtype=inputs.dtype)

        # size is batch, nclasses, 256, 256
        weights = 1.0 - torch.sum(target_one_hot, dim=tuple(range(2,len(inputs.shape))), keepdim=True)/torch.sum(target_one_hot)
        target_one_hot = weights*target_one_hot

        loss = self.ce(inputs, targets).unsqueeze(1) # shape is batch, 1, 256, 256
        loss = loss*target_one_hot

        size = targets.size(0)*targets.size(1)
        if len(inputs.shape)==5:
            size *= targets.size(2)
        return torch.sum(loss)/(torch.sum(weights)*size)