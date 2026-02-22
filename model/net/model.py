from torch import nn

from .encoder import CBDE
from .WDGB import WDGB


class WeatherNet(nn.Module):
    def __init__(self, cfg):
        super(WeatherNet, self).__init__()

        # Restorer
        self.R = WDGB(cfg)

        # Encoder
        self.E = CBDE(cfg)

    def forward(self, x_query, x_key):
        if self.training:
            fea, logits, labels, inter = self.E(x_query, x_key)

            restored = self.R(x_query, inter)

            return restored, logits, labels
        else:
            fea, inter = self.E(x_query, x_query)

            restored = self.R(x_query, inter)
            #修改点：输出例子
            #from utils.image_io import save_image_tensor
            #save_image_tensor(restored, '/data1/wangbin/AWDW-ReID/DW-ReID-weathernet/test.png')
            return restored
