
from collections import OrderedDict

import numpy as np
import torch

from models.Normalize import Normalize


def F(Si : list, sf : float, p : int=5) -> float:
    """Implementation of the aggregation function presented in the FaceQAN paper

    Args:
        Si (list): List of adversarial similarities.
        sf (float): Float presenting the symmetry similarity.
        p (int, optional): Exponent of the final power operation. Defaults to 5.

    Returns:
        float: Sample quality score.
    """
    return (((1. + np.mean(Si))/ 2.) * np.clip(1. - np.std(Si), 0., 1.) * sf)**p


def load_cosface():
    """Loads a pretrained CosFace ResNet100 model

    Returns:
        torch.nn.Module: Pretrained CosFace model used for evaluation in the FaceQAN paper
    """
    from models.iresnet import load_cosface as lc
    return lc()


def add_norm_to_model(model: torch.nn.Module, mean: list, std: list) -> torch.nn.Module:
    """Adds normalization to the top of given model, allowing for easy visualization of generated noise

    Args:
        model (torch.nn.Module): Given FR model.
        mean (list): Per channel mean values, for data normalization.
        std (list): Per channel deviation values, for data normalization.

    Returns:
        torch.nn.Module: Altered FR model, with added normalization layer on-top.
    """
    return torch.nn.Sequential(
        OrderedDict([
            ("norm_layer", Normalize(mean=np.array(mean), std=np.array(std))),
            ("base_model", model)
        ])
    )


def batch_n_noise(image: torch.Tensor, eps: float, batch_size: int) -> torch.Tensor:
    """Creates a batch of "batch_size" images and adds noise from a zero-centered uniform distribution with parameter "eps"
    
    Args:
        image (torch.Tensor): Input image in tensor form.
        eps (float): Controls the spread of the uniform distribution.
        batch_size (int): Number of copies to include in the batch.

    Returns:
        torch.Tensor: Batch of images with added uniform noise, ready for BIM attack.
    """
    img_temp = image.detach().clone()

    batch = torch.zeros((batch_size, *img_temp.shape)).cuda()

    for i in range(len(batch)):
        batch[i, :, :, :] = torch.clamp(img_temp + torch.FloatTensor(batch[i, :, :, :].shape).uniform_(-eps, eps).cuda(), -1., 1.) 

    return batch
