from typing import List, Tuple

import numpy as np
import torch
import yacs.config


def ricap(
    batch: Tuple[torch.Tensor, torch.Tensor], beta: float
) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[float]]]:
    data, targets = batch
    image_h, image_w = data.shape[2:]
    # get the custom ratio
    ratio = np.random.beta(beta, beta, size=2)
    # get new size
    w0, h0 = np.round(np.array([image_w, image_h]) * ratio).astype(np.int)
    w1, h1 = image_w - w0, image_h - h0

    # define a width and a height, defining the rest of three parts
    ws = [w0, w1, w0, w1]
    hs = [h0, h0, h1, h1]

    patches = []
    labels = []
    label_weights = []
    for w, h in zip(ws, hs):
        indices = torch.randperm(data.size(0))
        # randomly sample the length and width from uniform distribution
        x0 = np.random.randint(0, image_w - w + 1)
        y0 = np.random.randint(0, image_h - h + 1)
        patches.append(data[indices, :, y0:y0 + h, x0:x0 + w])
        labels.append(targets[indices])
        label_weights.append(h * w / (image_h * image_w))

    data = torch.cat(
        [torch.cat(patches[:2], dim=3),
         torch.cat(patches[2:], dim=3)], dim=2)
    targets = (labels, label_weights)

    return data, targets


class RICAPCollator:
    def __init__(self, config: yacs.config.CfgNode):
        self.beta = config.augmentation.ricap.beta

    def __call__(
        self, batch: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[float]]]:
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = ricap(batch, self.beta)
        return batch
