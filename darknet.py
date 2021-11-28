from typing import List

import torch
from torch import nn
import numpy as np

from blocks import first_darknet_layer, darknet_layer


class Darknet53(nn.Module):
    def __init__(
            self,
            path_to_weights: str or None = None,

    ):
        super().__init__()

        self.layers = [1, 2, 8, 8, 4]
        self.first_block = first_darknet_layer()
        self.layer1 = darknet_layer(32, 64, self.layers[0])
        self.layer2 = darknet_layer(64, 128, self.layers[1])
        self.layer3 = darknet_layer(128, 256, self.layers[2])
        self.layer4 = darknet_layer(256, 512, self.layers[3])
        self.layer5 = darknet_layer(512, 1024, self.layers[4])

        self.load_from_path(path_to_weights)
        if path_to_weights is not None:
            self.init_model()

    def load_from_path(self, path: str or None):
        pass

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.layer3(self.layer2(self.layer1(self.first_block(x))))
        x2 = self.layer4(x1)
        x3 = self.layer5(x2)
        return [x1, x2, x3]
