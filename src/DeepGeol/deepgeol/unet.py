  # coding: utf-8 -*-
#
# This file is part of DeepGeol.
#
# DeepGeol is free software: you can redistribute it and/or modify
# it under the terms of the GNU LLesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# DeepGeol is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DeepGeol.  If not, see <http://www.gnu.org/licenses/>.

"""
Copyright 2024-2025 Anthony Larcher
"""


__license__ = "LGPL"
__author__ = "Anthony Larcher"
__copyright__ = "Copyright 2024-2025 Anthony Larcher"
__maintainer__ = "Anthony Larcher"
__email__ = "anthony.larcher@univ-lemans.fr"
__status__ = "Production"
__docformat__ = "reS"


import torch

from .unet_parts import DoubleConv
from .unet_parts import Down2C
from .unet_parts import Up2C
from .unet_parts import OutConv
from .unet_parts import ASPP



class UNet(torch.nn.Module):
    """
    Flexible implementation of a U-Net architecture

    :param input_channels: int number of input channels, default is 1
    :param hidden_channels: list(int) number of convolution channels across the layers, default is [64, 128, 256, 512, 1024]
    :param n_classes: int number of output channels, default is 1
    :param dropout: bool if True, a dropout layer is added after each DoubleConv block, default is False
    :param batch_norm: bool if True a BatchNorm layer is added after each convolution,
        in this case the bias of the convolutional layer is turned to False
    :param bilinear: bool
    """
    def __init__(self,
                 input_channels=1,
                 hidden_channels=[64, 128, 256, 512, 1024],
                 n_classes=1,
                 dropout=False,
                 batch_norm=False,
                 bilinear=True,
                 attention=False
        ):
        super(UNet, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.n_classes = n_classes
        self.dropout= dropout
        self.batch_norm = batch_norm
        self.bilinear = bilinear
        self.innput_conv = None
        self.attention = attention

        self.inc = DoubleConv(input_channels, hidden_channels[0], dropout=dropout, batch_norm=batch_norm)

        # Encoder (Down)
        # 64 -> 128
        self.down0 = Down2C(hidden_channels[0], hidden_channels[1], dropout=dropout, batch_norm=batch_norm)
        # 128 -> 256
        self.down1 = Down2C(hidden_channels[1], hidden_channels[2], dropout=dropout, batch_norm=batch_norm)
        # 256 -> 512
        self.down2 = Down2C(hidden_channels[2], hidden_channels[3], dropout=dropout, batch_norm=batch_norm)
        # 512 -> 1024
        self.down3 = Down2C(hidden_channels[3], hidden_channels[4], dropout=dropout, batch_norm=batch_norm)

        self.aspp = ASPP(in_channels=hidden_channels[4], mid_channels=hidden_channels[4])

        # Decoder (Up)
        # On passe (Input venant du bas, Input venant du skip/Sortie)

        # up0 prend down3 (1024) et skip down2 (512)
        self.up0 = Up2C(hidden_channels[4], hidden_channels[3], bilinear=bilinear, attention=attention)

        # up1 prend up0 (512) et skip down1 (256)
        self.up1 = Up2C(hidden_channels[3], hidden_channels[2], bilinear=bilinear, attention=attention)

        # up2 prend up1 (256) et skip down0 (128)
        self.up2 = Up2C(hidden_channels[2], hidden_channels[1], bilinear=bilinear, attention=attention)

        # up3 prend up2 (128) et skip inc (64)
        self.up3 = Up2C(hidden_channels[1], hidden_channels[0], bilinear=bilinear, attention=attention)

        self.outc = OutConv(hidden_channels[0], n_classes)

        # J'ai mis en commentaires car dans ma boucle d'apprentissage
        # j'utilise la fonction de loss BCEWithLogitsLoss qui applique une sigmoid à la sortie du réseau
        # self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        down_outputs = []

        down_outputs.append(self.inc(x))

        down_outputs.append(self.down0(down_outputs[-1]))
        down_outputs.append(self.down1(down_outputs[-1]))
        down_outputs.append(self.down2(down_outputs[-1]))
        down_outputs.append(self.down3(down_outputs[-1]))

        # On applique notre module de contextualisation ASPP sur la sortie du dernier bloc de down
        # C'est le bottleneck du U-Net
        down_outputs[-1] = self.aspp(down_outputs[-1])
        x = self.up0(down_outputs[-1], down_outputs[-2])
        x = self.up1(x, down_outputs[-3])
        x = self.up2(x, down_outputs[-4])
        x = self.up3(x, down_outputs[-5])

        logits = self.outc(x)
        out = logits
        # out = self.sigmoid(logits)
        return out

# def test_unet():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Device: {device}")
#     model = UNet().to(device)
#     x = torch.randn(1, 1, 256, 256).to(device)  # Batch size of 1, 1 input channel, and spatial dimensions of 572x572
#     output = model(x)
#     print(f"Output shape: {output.shape}")  # Should be (1, n_classes, H_out, W_out)
