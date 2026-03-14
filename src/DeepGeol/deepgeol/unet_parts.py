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
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

# from .resblock import ResBlock
# from .attention_mechanisms import AttentionGateModule
# from .attention_mechanisms import ContextAttentionBlock


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=None, batch_norm=False):
        """
        (convolution => [BN] => ReLU) * 2 + DropOut [optional]
    
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param mid_channels: number of middle channels, if None, mid-channels = out_channels, default is None
        :type mid_channels: int
        :param dropout: value of the DropOut probability, if None, Drop-Out is not applied, default is None
        :type dropout: float
        :param batch_norm: apply BatchNorm2d after each convolution layer, default is False
        :type batch_norm: bool
        """
        super().__init__()

        # Apply the rule in function comment
        if not mid_channels:
            mid_channels = out_channels

        # we must split our sequential declaration due to the paramaters conditions
        # like batch_norm and dropout
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)))

        if batch_norm:
            self.double_conv.append(nn.BatchNorm2d(mid_channels))

        self.double_conv.append(nn.ReLU(inplace=True))
        self.double_conv.append(nn.Conv2d(mid_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)))

        if batch_norm:
            self.double_conv.append(nn.BatchNorm2d(mid_channels))

        self.double_conv.append(nn.ReLU(inplace=True))

        if dropout is not None:
            self.double_conv.append(nn.Dropout2d(dropout, inplace=False))

        
    def forward(self, x):
        return self.double_conv(x)


class Down2C(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=False, batch_norm=False):
        """
        Downscaling with maxpool then double conv block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param mid_channels: number of middle channels, if None, mid-channels = out_channels, default is None
        :type mid_channels: int
        :param dropout: value of the DropOut probability, if None, Drop-Out is not applied, default is None
        :type dropout: float
        :param batch_norm: apply BatchNorm2d after each convolution layer, default is False
        :type batch_norm: bool
        """
        super().__init__()
        self.down_2c = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0), dilation=(1,1), ceil_mode=False),
            DoubleConv(in_channels, out_channels, mid_channels, dropout, batch_norm)
        )

    def forward(self, x):
        return self.down_2c(x)


class Up2C(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        :param in_channels: Canaux de l'input venant du bas (couche précédente)
        :param out_channels: Canaux de la skip connection (et donc de la sortie du bloc)
        """
        super().__init__()

        # Si bilinear, on upsample simplement la géométrie, les canaux ne changent pas lors de l'upsample
        if bilinear:
            self.up = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
            # Après concaténation, on aura in_channels + out_channels
            # On réduit ensuite via DoubleConv
            self.conv = DoubleConv(in_channels + out_channels, out_channels, mid_channels=out_channels)

        # Si Transpose, c'est une couche apprenante qui change les canaux
        else:
            # On suppose que Transpose réduit les canaux par 2 (ex: 1024 -> 512)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # Après concaténation: (in_channels // 2) venant du up + out_channels venant du skip
            self.conv = DoubleConv((in_channels // 2) + out_channels, out_channels)

    def forward(self, x1, x2):
        #print(f'x1 avant : {x1.shape}')
        # x1 is the input from the previous down layer
        # x2 is the input from the previous skip connection
        # for exemple x1 is h//4 w//4 and x2 is h//2 w//2
        x1 = self.up(x1)

        #print(f'x1 apres : {x1.shape}')
        #print(f'x2 : {x2.shape}')
        
        # We must take care of the padding

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # input is CHW
        # Concatenate the two inputs (x1 has already been processed)
        x = torch.cat([x2, x1], dim=1)

        # Apply a 2D convolution
        output = self.conv(x)
        
        return output


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Apply a last 2D convolution to produce the final output

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        return self.conv(x)


def test():
    """
    Test the different blocks of the UNet architecture

    Returns:
        None
    """

    # Test DoubleConv
    x = torch.randn(1, 3, 256, 256)
    double_conv = DoubleConv(3, 64, dropout=0.5, batch_norm=True)
    y = double_conv(x)
    print("DoubleConv output shape:", y.shape)

    # Test Down2C
    down2c = Down2C(3, 64, dropout=0.5, batch_norm=True)
    y = down2c(x)
    print("Down2C output shape:", y.shape)

    # Test Up2C
    x1 = torch.randn(1, 128, 128, 128)  # Input from the previous layer
    x2 = torch.randn(1, 64, 256, 256)   # Input from the skip connection
    up2c = Up2C(192, 64)  # in_channels is the sum of channels from x1 and x2
    y = up2c(x1, x2)
    print("Up2C output shape:", y.shape)

    # Test OutConv
    out_conv = OutConv(64, 1)
    y = out_conv(y)
    print("OutConv output shape:", y.shape)