# ASPP — Pense-bête complet

> Module Atrous Spatial Pyramid Pooling intégré au bottleneck du UNet pour la segmentation géologique.

---

## 1. Pourquoi en a-t-on besoin ?

Une convolution classique ne voit qu'une seule échelle à la fois. Or une faille géologique peut être :
- **fine** → détectable à petite échelle (quelques pixels)
- **large** → détectable à grande échelle (dizaines de pixels)

Avec une seule convolution, tu rates forcément l'une des deux. L'ASPP analyse la même feature map à **plusieurs échelles simultanément** et fusionne les résultats.

---

## 2. Le concept clé : la convolution dilatée (atrous)

Une Conv 3×3 avec `dilation=2` ne regarde pas 3 pixels consécutifs — elle **saute** de 2 pixels entre chaque point du kernel.

```
dilation=1  → X X X     champ réceptif = 3 pixels
              X X X
              X X X

dilation=2  → X . X . X     champ réceptif = 5 pixels
              . . . . .      (sans paramètres supplémentaires)
              X . X . X
              . . . . .
              X . X . X

dilation=3  → X . . X . . X     champ réceptif = 7 pixels
```

**Règle générale :** champ réceptif = `(kernel_size - 1) × dilation + 1`

Pour que la sortie garde la même taille spatiale que l'entrée :
```
padding = dilation    (pour un kernel 3×3)
padding = 0           (pour un kernel 1×1)
```

---

## 3. Architecture des 5 branches

```
                    Feature map x (B, C_in, H, W)
                            │
        ┌───────────┬───────┴────────┬───────────┬──────────────┐
        │           │                │            │              │
   Conv 1×1     Conv 3×3         Conv 3×3     Conv 3×3     AvgPool(1,1)
   dil=1        dil=1            dil=2        dil=3        + Conv 1×1
   pad=0        pad=1            pad=2        pad=3        + F.interpolate
        │           │                │            │              │
(B,mid,H,W) (B,mid,H,W)      (B,mid,H,W) (B,mid,H,W)    (B,mid,H,W)
        │           │                │            │              │
        └───────────┴────────────────┴────────────┴──────────────┘
                                     │
                               torch.cat dim=1
                            (B, mid × 5, H, W)
                                     │
                             conv_1x1_out_1
                            (B, mid_ch, H, W)
```

| Branche | kernel | dilation | padding | Ce qu'elle capte |
|---------|--------|----------|---------|-----------------|
| 1 | 1×1 | 1 | 0 | features pixel par pixel, sans contexte |
| 2 | 3×3 | 1 | 1 | contexte local immédiat |
| 3 | 3×3 | 2 | 2 | structures à moyenne échelle |
| 4 | 3×3 | 3 | 3 | structures à grande échelle |
| 5 | AvgPool | — | — | contexte global de l'image entière |

---

## 4. Pourquoi `F.interpolate` sur la branche 5 ?

`AdaptiveAvgPool2d((1,1))` compresse toute la feature map en **un seul pixel** par canal :

```
(B, C_in, H, W)  →  AdaptiveAvgPool2d  →  (B, mid_ch, 1, 1)
```

Problème : `torch.cat` exige que **toutes les branches aient la même taille spatiale**.
Tu ne peux pas concaténer `(B, C, H, W)` avec `(B, C, 1, 1)`.

`F.interpolate` "étale" ce pixel unique sur toute la grille :

```
(B, mid_ch, 1, 1)  →  F.interpolate(size=(H,W))  →  (B, mid_ch, H, W)
```

Chaque pixel de la grille reçoit la même valeur — c'est la moyenne globale de l'image, disponible en tout point de la feature map.

---

## 5. Implémentation complète

```python
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (Chen et al., DeepLab v3, 2017).

    Applies 5 parallel branches on the same input feature map to
    capture multi-scale context. Designed to be inserted at the
    bottleneck of the UNet (smallest spatial resolution).

    Args:
        in_channels : input feature map channels (= hidden_channels[-1])
        mid_channels: channels produced by each branch and at output
    """
    def __init__(self, in_channels, mid_channels=256):
        super().__init__()

        # Branch 1 : Conv 1×1 — pixel-level features, no spatial context
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                      stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 2 : Conv 3×3 dilation=1 — standard local context
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                      stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 3 : Conv 3×3 dilation=2 — medium-scale context
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                      stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 4 : Conv 3×3 dilation=3 — large-scale context
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                      stride=1, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 5 : Global Average Pooling → (B, mid_ch, 1, 1)
        # Captures the full image context in a single value per channel.
        # F.interpolate in forward() restores spatial dimensions for cat.
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion : merge 5 branches (mid_channels × 5) back to mid_channels
        self.conv_1x1_out_1 = nn.Sequential(
            nn.Conv2d(mid_channels * 5, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)                                          # (B, mid, H, W)
        b2 = self.branch2(x)                                          # (B, mid, H, W)
        b3 = self.branch3(x)                                          # (B, mid, H, W)
        b4 = self.branch4(x)                                          # (B, mid, H, W)

        # Global branch : pool to (B, mid, 1, 1) then upsample to (B, mid, H, W)
        # F.interpolate is mandatory here: torch.cat requires identical spatial dims
        b5 = self.branch5(x)
        b5 = F.interpolate(b5, size=x.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate all branches along channel dimension → (B, mid*5, H, W)
        out = torch.cat([b1, b2, b3, b4, b5], dim=1)

        # Fuse back to mid_channels → (B, mid, H, W)
        return self.conv_1x1_out_1(out)
```

---

## 6. Intégration dans le UNet

L'ASPP se place au **bottleneck** — après le dernier `down`, avant le premier `up`.

```
Encodeur          Bottleneck         Décodeur
─────────────────────────────────────────────
inc (256×256)
down0 (128×128)
down1 (64×64)
down2 (32×32)
down3 (16×16)  →  ASPP (16×16)  →  up0 (32×32)
                                    up1 (64×64)
                                    up2 (128×128)
                                    up3 (256×256)
```

**Pourquoi le bottleneck ?**
- La feature map est à **16×16** — les convolutions dilatées coûtent très peu
- C'est le point où le réseau a le plus besoin de contexte multi-échelle
- Tout ce que l'ASPP capture se propage ensuite dans tout le décodeur

**Dans `unet.py` :**

```python
# Dans __init__ :
self.down3 = Down2C(hidden_channels[3], hidden_channels[4], ...)
self.aspp  = ASPP(in_channels=hidden_channels[4],
                  mid_channels=hidden_channels[4])   # sortie = même nb de canaux

# Dans forward :
down_outputs.append(self.down3(down_outputs[-1]))
down_outputs[-1] = self.aspp(down_outputs[-1])       # enrichit le bottleneck

x = self.up0(down_outputs[-1], down_outputs[-2])     # décodeur inchangé
```

**Important :** l'ASPP doit produire le **même nombre de canaux** que `down3`
(`hidden_channels[4]` = 512) pour que `up0` reçoive ce qu'il attend.
Ne pas mettre de `conv_1x1_out` final (celui qui projette à 1 canal) dans cette version.

---

## 7. Référence

> Chen et al., **"Rethinking Atrous Convolution for Semantic Image Segmentation"**
> (DeepLab v3), arXiv 2017.
> https://arxiv.org/abs/1706.05587