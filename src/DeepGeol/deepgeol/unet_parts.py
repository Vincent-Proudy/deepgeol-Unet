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
        super().__init__()

        # Création de notre reseau de convolution avec plus ou moins de bloc selon les paramètres

        if not mid_channels:
            mid_channels = out_channels

        # On a besoin de faire un sequential pour pouvoir faire append les différentes couches car on ne sait
        # pas à l'avance combien de couches on va ajouter (batch norm et dropout sont optionnels). C'est plus
        # simple que de faire des if pour chaque type de configuration
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        )

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
    """
    Un bloc d'encodeur du Unet
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=False, batch_norm=False):
        super().__init__()

        self.down_2c = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0), dilation=(1,1), ceil_mode=False),
            DoubleConv(in_channels, out_channels, mid_channels, dropout, batch_norm)
        )

    def forward(self, x):
        return self.down_2c(x)


class Up2C(nn.Module):
    """
    Un bloc de déecodeur du Unet
    """
    def __init__(self, in_channels, out_channels, bilinear=True, attention=False):
        super().__init__()
        self.use_attention = attention

        # x1 sera notre tenseur qu'on devra upsamplé pour le faire correspondre à x2 (skip connection)
        # Pour ça, on a deux moyens :
        # - soit on utilise un upsamble mathématique : Chaque nouveau pixel est calculé par une moyenne
        #   pondérée de ses voisin
        # - soit on utilise une convolution qui fera l'upsampling et qui va apprendre a le faire :
        #   c'est la convolution transpose (ConvTranspose2d)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
            # après l'upsampling, on concatène x1 et x2, donc le nombre de canaux d'entrée de la
            # double conv est la somme des canaux de x1 et x2
            self.conv = DoubleConv(in_channels + out_channels, out_channels, mid_channels=out_channels)
            # Comme le bilinear ne change pas le nombre de canaux de x1, on peut faire une affectation
            # directe sans toucher au nombre de canaux
            F_g = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # après l'upsampling, on concatène x1 et x2, donc le nombre de canaux d'entrée de la
            # double conv est la somme des canaux de x1 et x2 mais comme la conv transpose réduit
            # le nombre de canaux de x1 de moitié, on fera attention
            self.conv = DoubleConv((in_channels // 2) + out_channels, out_channels, batch_norm=False)
            # Par contre là, on doit faire attention au nombre de canaux de x1 après l'upsampling,
            # qui est réduit de moitié par la conv transpose. (Ca change du bilinear)
            F_g = in_channels // 2

        if attention:
            # Quand on utilise le mécanisme d'attention, on doit créer une instance
            # pour le forward, logique
            self.att = AttentionGate(F_g=F_g, F_x=out_channels, F_int=out_channels // 2)

    def forward(self, x1, x2):
        # x1 le tenseur d'entrée de la couche précédente (couche du bas) qui doit être upsamplé
        # pour correspondre à x2 (skip connection)
        # x2 le tenseur de la skip connection (couche du haut) qui doit être concaténé à x1 après upsampling
        #
        # Un exemple des dimensions de x1 et x2 :
        # x1 : (B, 512, 16, 16)  # couche du bas
        # x2 : (B, 256, 32, 32)  # skip connection

        # on upsample le tenseur selon la méthode choisie (bilinear ou conv transpose)
        # Dans tous les cas, ca fait la meme chose, juste avec la conv transpose, comme
        # c'est une couche a par entière, on peut avoir de l'overfit sur les petits dataset,
        # ou des artefacts si c'est mal régularisé, alors que le bilinear est plus stable mais moins performant
        x1 = self.up(x1)

        # On calcul le filtre d'attention pour x2 venant de la connexion résiduelle
        if self.use_attention:
            x2 = self.att(g=x1, x=x2)
        
        # Il faut aussi gérer le padding car les dimensions de x1 et x2 peuvent ne pas correspondre
        # à cause des arrondis lors des maxpooling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # ajoute des 0 autour de x1 pour atteindre du tenseur x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # On peut concatené les deux tenseurs en toute sérénité car on est sur que c'est
        # ok pour les dimensions
        x = torch.cat([x2, x1], dim=1)

        # Reste plus qu'a faire une convolution sur notre image final
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

    Args:
        F_g  : nombre de canaux du signal de gating (ça vient du décodeur x1)
        F_x  : nombre de canaux de la connexion résiduelle (ça vient de l'encodeur x2)
        F_int: nombre de canaux intermédiaires (par ex. F_x // 2), espace commun où
               g et x sont comparés pour calculer les coefficients d'attention.
    """
    def __init__(self, F_g, F_x, F_int):
        super(AttentionGate, self).__init__()

        # Ok, là le but ca va être d'amener g (la sortie du down) et x (la connexion résiduelle) dans
        # un espace commun de dimension F_int pour pouvoir les comparer et calculer les coefficients d'attention.
        # Car si on le fait directement dans leur espace d'origine (dimensions différentes), ca n'aurait pas de sens

        # Ici c'est pour amener g dans l'espace commun (dim=F_int))
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        # Ici c'est pour amener x dans l'espace commun (dim=F_int)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        # Après ça, on peut calculer nos coeff d'attention, qui sera entre 0 et 1 grâce a la signoide.
        # la dimension de sortie sera avec 1 canal qui stokera les coeff .
        #
        # Je crois ce canal s'appelle la carte d'attention dans le papier
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g1 et x1, les g et x dans l'espace commun
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # On a un problème car g1 a la résolution du décodeur (ex: 16×16)
        # et x1 a la résolution de l'encodeur (ex: 32×32)
        # On peut pas additionner deux tenseurs de tailles différentes.
        # F.interpolate remonte g1 à la taille de x1 par interpolation bilinéaire
        # size=x1.shape[2:] récupère (H_x, W_x) — les deux dernières dimensions
        # (B, F_int, 16, 16) → (B, F_int, 32, 32). c'est un peu le meme esprit que dans
        # le bloc Up2c
        g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        # On calcul la carte d'attention
        # dim de alpha = (B, 1, H_x, W_x), en gros une seule carte pour tous les canaux
        alpha = self.psi(self.relu(g1 + x1))

        # Enfin, on fait le filtrage sur la connexion résiduelle x en multipliant chaque
        # pixel de x par son coefficient d'attention alpha, comme ça on donne plus ou moins
        # d'importance à chaque pixel selon le contexte donné par g. Si on veut rapprocher
        # à ça a notre cas :
        # - un pixel qui a une valeur proche de 1, alors on le conserve car info utile (par ex. faille)
        # - un pixel qui a une valeur proche de 0, alors on le supprime car pas d'info utile (par ex. roche)
        return x * alpha

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (Chen et al., DeepLab v3, 2017).

    Ce module est pratique au niveau du bottleneck du Unet (plus bas dans l'encodeur/down)
    pour capturer des features à différentes échelles. Dans mon cas, on a 5 branches,
    avec chacune un rôle différent pour choper du contexte à différentes échelles.
    """
    def __init__(self, in_channels, mid_channels=256):
        super().__init__()

        # Première branche simple, une convolution sans contexte (on regarde pas les voisins du pixel)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Deuxième branche, convolution standard de 3par3, qui regarde les 8 voisins du pixel.
        # On commence a augmenter le contexte mais c'est encore assez local
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Troisième branche, convolution dilatée de 3par3 avec dilation=2, qui regarde les voisins
        # à distance 2 du pixel, on augmente encore le contexte, on regarde plus loin que les 8 voisins immédiats,
        # mais on perd un peu de précision locale
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Quatrième branche, convolution dilatée de 3par3 avec dilation=3, qui regarde les voisins à distance 3 du pixel,
        # on regarde encore plus loin. On est très mauvais en local mais on a un gros contexte
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Cinquième branche, pooling global, qui regarde tout le contexte de l'image. C'est la branche qui a le
        # plus de contexte mais aucune précision locale. On va faire un pooling global pour réduire à une seule valeur
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Après, on concatène les 5 branches, ce qui nous donne un tenseur de dimension (B, mid_channels*5, H, W)
        self.conv_1x1_out_1 = nn.Sequential(
            nn.Conv2d(mid_channels * 5, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Enfin, on peut faire une dernière convolution 1x1 pour réduire à 1 canal si on veut faire de la
        # segmentation binaire, ou garder mid_channels si c'est pour le bottleneck du Unet
        self.conv_1x1_out = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x):
        # Nos branches de convulution avec plus ou moins de contexte
        # A noter que chaque branche à comme dim de sortie (B, mid_channels, H, W)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # (B, mid, 1, 1)
        b5 = self.branch5(x)
        # F.interpolate obligatoire ici car cat() exige la même taille .
        # comme b5 est (B, mid, 1, 1) après le pooling, ça l'étale sur (H, W).
        # Chaque pixel reçoit la même valeur
        b5 = F.interpolate(b5, size=x.shape[2:], mode='bilinear', align_corners=True)

        # Ensuite quand on sait que les dimensions sont ok, on peut fusionner les
        # 5 branches pour avoir un tenseur de dim (B, mid_channels*5, H, W)
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)

        # Et enfin, on peut faire une convolution 1x1 pour réduire à mid_channels ou 1 canal selon l'usage
        # Comme nous on est dans le bottleneck du Unet, on va garder mid_channels pour pas perdre d'info,
        # mais si c'était pour faire la dernière couche de sortie, on ferait la conv 1x1 pour réduire à 1 canal
        out = self.conv_1x1_out_1(out)
        # out = self.conv_1x1_out(out)

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