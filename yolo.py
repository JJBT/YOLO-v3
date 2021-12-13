from torch import nn

from darknet import Darknet53


class YOLO(nn.Module):
    def __init__(self):
        super().__init__()
        self.darknet = Darknet53()

