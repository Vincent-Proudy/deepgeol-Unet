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
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False))

        if batch_norm:
            self.double_conv.append(nn.BatchNorm2d(mid_channels))

        self.double_conv.append(nn.ReLU(inplace=True))
        self.double_conv.append(nn.Conv2d(mid_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False))

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
    def __init__(self, in_channels, out_channels, bilinear=True, attention=False):
        """
        :param in_channels: Canaux de l'input venant du bas (couche précédente)
        :param out_channels: Canaux de la skip connection (et donc de la sortie du bloc)
        """
        super().__init__()
        self.use_attention = attention

        # Si bilinear, on upsample simplement la géométrie, les canaux ne changent pas lors de l'upsample
        if bilinear:
            self.up = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
            # Après concaténation, on aura in_channels + out_channels
            # On réduit ensuite via DoubleConv
            self.conv = DoubleConv(in_channels + out_channels, out_channels, mid_channels=out_channels)
            F_g = in_channels  # gating signal channels (before upsample)

        # Si Transpose, c'est une couche apprenante qui change les canaux
        else:
            # On suppose que Transpose réduit les canaux par 2 (ex: 1024 -> 512)
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # Après concaténation: (in_channels // 2) venant du up + out_channels venant du skip
            self.conv = DoubleConv((in_channels // 2) + out_channels, out_channels, batch_norm=False)
            F_g = in_channels // 2

        if attention:
            # F_x = skip connection channels, F_int = intermediate (half of skip)
            self.att = AttentionGate(F_g=F_g, F_x=out_channels, F_int=out_channels // 2)

    def forward(self, x1, x2):
        #print(f'x1 avant : {x1.shape}')
        # x1 is the input from the previous down layer
        # x2 is the input from the previous skip connection
        # for exemple x1 is h//4 w//4 and x2 is h//2 w//2
        x1 = self.up(x1)

        if self.use_attention:
            x2 = self.att(g=x1, x=x2)
        
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), bias=False)

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al., 2018 - Attention U-Net).

    Filters the skip connection x using a gating signal g from the decoder,
    producing a spatial attention map that highlights relevant regions.
    This helps the model focus on geological structures (faults, layers)
    while suppressing irrelevant background features.

    Args:
        F_g  : number of channels in the gating signal (from decoder)
        F_x  : number of channels in the skip connection (from encoder)
        F_int: number of intermediate channels (typically F_x // 2)
    """
    def __init__(self, F_g, F_x, F_int):
        super(AttentionGate, self).__init__()

        # Project gating signal to intermediate space
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        # Project skip connection to intermediate space
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        # Produce a single-channel spatial attention map in [0, 1]
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: gating signal from the decoder (lower resolution)
            x: skip connection from the encoder (higher resolution)
        Returns:
            x weighted by the attention map (same shape as x)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Upsample g to match x spatial dimensions before addition
        g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        # Combine, activate, then compute attention coefficients
        alpha = self.psi(self.relu(g1 + x1))

        # Weight the skip connection pixel-wise
        return x * alpha

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (Chen et al., DeepLab v3, 2017).

    Applies 5 parallel branches on the same input feature map:
      - 1 Conv 1x1 (local features, no context)
      - 3 Conv 3x3 with increasing dilation (multi-scale context)
      - 1 Global Average Pooling (full image context)
    All branches are concatenated then fused by two Conv 1x1.

    Args:
        in_channels : input feature map channels
        mid_channels: channels produced by each branch (middle_channel in the image)
    """
    def __init__(self, in_channels, mid_channels=256):
        super().__init__()

        # Conv 1x1, kernel=1, dilation=1, padding=0
        # Captures pixel-level features without any spatial context
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Conv 3x3, dilation=1, padding=1
        # Standard convolution — receptive field = 3 pixels
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Conv 3x3, dilation=2, padding=2
        # Dilated convolution — receptive field = 5 pixels
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Conv 3x3, dilation=3, padding=3
        # Dilated convolution — receptive field = 7 pixels
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Global Average Pooling → (B, in_channels, 1, 1)
        # Captures full image context in a single value per channel.
        # Followed by Conv 1x1 to project to mid_channels,
        # then F.interpolate to restore spatial size before concat.
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # fuse the 5 concatenated branches
        # 5 branches × mid_channels → mid_channels
        self.conv_1x1_out_1 = nn.Sequential(
            nn.Conv2d(mid_channels * 5, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # final projection to 1 output channel
        self.conv_1x1_out = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x) # (B, mid, H, W)
        b2 = self.branch2(x) # (B, mid, H, W)
        b3 = self.branch3(x) # (B, mid, H, W)
        b4 = self.branch4(x) # (B, mid, H, W)

        # Global branch : pool → (B, mid, 1, 1) then upsample → (B, mid, H, W)
        # F.interpolate is needed because torch.cat requires identical spatial dims
        b5 = self.branch5(x)
        b5 = F.interpolate(b5, size=x.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate all branches : (B, mid*5, H, W)
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)

        out = self.conv_1x1_out_1(out)   # (B, mid, H, W)
        # On our case, i'm going to apply the ASPP on the bottleneck,
        # so i can't apply the last conv 1x1 to reduce to 1 channel,
        # because the output of the ASPP will be the input of the first up block,
        # which expect mid_channels as input channels.
        # out = self.conv_1x1_out(out)     # (B, 1, H, W)

        return out

# def test():
#     """
#     Test the different blocks of the UNet architecture
#
#     Returns:
#         None
#     """
#
#     # Test DoubleConv
#     x = torch.randn(1, 3, 256, 256)
#     double_conv = DoubleConv(3, 64, dropout=0.5, batch_norm=True)
#     y = double_conv(x)
#     print("DoubleConv output shape:", y.shape)
#
#     # Test Down2C
#     down2c = Down2C(3, 64, dropout=0.5, batch_norm=True)
#     y = down2c(x)
#     print("Down2C output shape:", y.shape)
#
#     # Test Up2C
#     x1 = torch.randn(1, 128, 128, 128)  # Input from the previous layer
#     x2 = torch.randn(1, 64, 256, 256)   # Input from the skip connection
#     up2c = Up2C(192, 64)  # in_channels is the sum of channels from x1 and x2
#     y = up2c(x1, x2)
#     print("Up2C output shape:", y.shape)
#
#     # Test OutConv
#     out_conv = OutConv(64, 1)
#     y = out_conv(y)
#     print("OutConv output shape:", y.shape)