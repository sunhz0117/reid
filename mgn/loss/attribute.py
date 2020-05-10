#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

from IPython import embed

class AttributeLoss(nn.Module):
    def __init__(self):
        super(AttributeLoss, self).__init__()

        # 2 labels
        self.gender = nn.CrossEntropyLoss(ignore_index=-1)
        self.hair = nn.CrossEntropyLoss(ignore_index=-1)
        self.up = nn.CrossEntropyLoss(ignore_index=-1)
        self.down = nn.CrossEntropyLoss(ignore_index=-1)
        self.clothes = nn.CrossEntropyLoss(ignore_index=-1)
        self.hat = nn.CrossEntropyLoss(ignore_index=-1)
        self.backpack = nn.CrossEntropyLoss(ignore_index=-1)
        self.bag = nn.CrossEntropyLoss(ignore_index=-1)
        self.hair = nn.CrossEntropyLoss(ignore_index=-1)
        self.handbag = nn.CrossEntropyLoss(ignore_index=-1)

        # 4 labels
        self.age = nn.CrossEntropyLoss(ignore_index=-1)

        # 8 labels: upblack, upwhite, upred, uppurple, upyellow, upgray, upblue, upgreen
        self.upcolor = nn.CrossEntropyLoss(ignore_index=-1)

        # 9 labels: downblack, downwhite, downpink, downpurple, downyellow, downgray, downblue, downgreen, downbrown
        self.downcolor = nn.CrossEntropyLoss(ignore_index=-1)


    def forward(self, inputs, attribute):

        # embed()
        losses = [
            self.gender(inputs[:, 0:2], attribute[:, 0]),
            self.hair(inputs[:, 2:4], attribute[:, 1]),
            self.up(inputs[:, 4:6], attribute[:, 2]),
            self.down(inputs[:, 6:8], attribute[:, 3]),
            self.clothes(inputs[:, 8:10], attribute[:, 4]),
            self.hat(inputs[:, 10:12], attribute[:, 5]),
            self.backpack(inputs[:, 12:14], attribute[:, 6]),
            self.bag(inputs[:, 14:16], attribute[:, 7]),
            self.handbag(inputs[:, 16:18], attribute[:, 8]),
            self.age(inputs[:, 18:22], attribute[:, 9]),
            self.upcolor(inputs[:, 22:30], attribute[:, 10]),
            self.downcolor(inputs[:, 30:39], attribute[:, 11]),
        ]

        return sum(losses) / len(losses)
