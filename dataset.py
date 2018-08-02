import os

import torch.utils.data as data
from torchvision.datasets import ImageFolder


def get_loader(root, transform, is_train, batch_size=32):
    if is_train:
        root = os.path.join(root, "train")
    else:
        root = os.path.join(root, "test")

    dataset = ImageFolder(root, transform)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=6)
    return loader
