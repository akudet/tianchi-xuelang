import torch.nn as nn

import torchvision.models as models


class BaselineModel(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.body = nn.Sequential(*modules)
        self.head = nn.Linear(2048, 48)

    def forward(self, x):
        x = self.body(x)
        x = x.view(-1, self.head.in_features)
        x = self.head(x)
        return x
