from typing import List, Tuple

import numpy as np
import torch
import yacs.config


def cutmix(
    batch: Tuple[torch.Tensor, torch.Tensor], alpha: float
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    data, targets = batch

    # random permutation from 0 to batch - 1
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    # alpha is scalar, np.random.beta returns a sample from beta distribution(alpha, alpha)
    # range: [0, 1]
    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    # uniform distribution from 0 to image width/image height, set up the center of the substitute cx, cy
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    # keep ratio of width to height
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    # replace data in random area with different data in the same area
    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets


class CutMixCollator:
    def __init__(self, config: yacs.config.CfgNode):
        self.alpha = config.augmentation.cutmix.alpha

    def __call__(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
        # default_collate: Puts each data field into a tensor with outer dimension batch size
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = cutmix(batch, self.alpha)
        return batch
