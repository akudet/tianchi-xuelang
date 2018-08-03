import torch.nn as nn

import torchvision.models as models


class BaselineModel(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.body = nn.Sequential(
            nn.Sequential(*modules),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 48),
        )

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)
