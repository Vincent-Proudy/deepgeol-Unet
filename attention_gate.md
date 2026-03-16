# Attention Gate — Pense-bête complet

> Mécanisme implémenté dans le cadre du TP UNet pour la segmentation géologique (failles).

---

## 1. Pourquoi en a-t-on besoin ?

Dans un UNet **sans** attention, les **skip connections** transmettent **toutes** les features de l'encodeur au décodeur — y compris les régions de fond (roche banale, bruit, textures irrelevantes).

Le décodeur doit donc apprendre à ignorer ce bruit seul, ce qui est difficile avec seulement 600 images.

**Problème concret :** sur une image géologique, 90–95 % des pixels sont du fond. Sans filtre, les skip connections inondent le décodeur de features inutiles.

---

## 2. Principe général

```
skip connection (encodeur)  ──────────────────────────────────► ✕ ──► décodeur
                                          ▲                     ▲
                                          │                     │
gating signal (décodeur bas) ──► AG ──► carte α (0→1) ─────────┘
```

L'Attention Gate produit une **carte d'attention α** (même taille spatiale que la skip connection), avec des valeurs entre 0 et 1.

La skip connection est ensuite **multipliée pixel par pixel** par α :
- Zone faille → α ≈ 0.9 → features conservées
- Zone fond   → α ≈ 0.05 → features écrasées

On obtient ce résultat grâce à un signoïde, car si on a une activation forte dans le décodeur, 
on veut que la skip connection soit conservée. Si l'activation est faible, on veut qu'elle soit filtrée.

---

## 3. Les deux entrées de l'Attention Gate

| Signal | Nom dans le code | Origine | Ce qu'il contient |
|--------|-----------------|---------|-------------------|
| `x` | skip connection | **Encodeur** (`down0`, `down1`…) | Précis, haute résolution, **pas de contexte global** |
| `g` | gating signal   | **Décodeur** (bloc du niveau inférieur) | Flou, basse résolution, **contexte global** (a traversé le bottleneck) |

**Clé de compréhension :**
- L'encodeur "voit" les détails fins mais ne sait pas si c'est une faille ou du bruit
- Le décodeur "sait" approximativement où chercher mais a perdu la précision
- **Ensemble**, ils produisent une carte qui dit "ici oui, là non"

---

## 4. Étapes de calcul

```
x ──► W_x (Conv 1×1 + BN) ──────────────┐
                                         ▼
                                        (+) ──► ReLU ──► ψ (Conv 1×1 + BN + Sigmoid) ──► α
                                         ▲
g ──► W_g (Conv 1×1 + BN) ──► upsample ─┘

résultat : x_filtré = x × α
```

### En code PyTorch :
```python
def forward(self, g, x):
    g1 = self.W_g(g)                                          # projette g
    x1 = self.W_x(x)                                          # projette x
    g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear') # aligne les résolutions
    alpha = self.psi(self.relu(g1 + x1))                      # carte d'attention
    return x * alpha                                           # filtre la skip
```

---

## 5. Où se place l'AG dans le UNet

```
Encodeur                    Décodeur
─────────────────────────────────────────────────────

down0 (128×128, 64ch)  ──x──► AG ◄──g── up2 (128×128, 64ch)
                                │
                                ▼
                          skip filtrée ──► concaténation ──► up2

down1 (64×64, 128ch)   ──x──► AG ◄──g── up1 (64×64, 128ch)
down2 (32×32, 256ch)   ──x──► AG ◄──g── up0 (32×32, 256ch)

                        [bottleneck 16×16, 512ch]
```

Le `g` de chaque AG vient **du bloc up situé un niveau en dessous** — celui qui a plus de contexte.

---

## 6. Implémentation dans le code

### Dans `unet_parts.py` — la classe `AttentionGate`

```python
class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al., 2018 - Attention U-Net).
    Filtre la skip connection x avec le gating signal g du décodeur.

    F_g  : canaux du gating signal (depuis le décodeur)
    F_x  : canaux de la skip connection (depuis l'encodeur)
    F_int: canaux intermédiaires (typiquement F_x // 2)
    """
    def __init__(self, F_g, F_x, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_x, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1    = self.W_g(g)
        x1    = self.W_x(x)
        g1    = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        alpha = self.psi(self.relu(g1 + x1))
        return x * alpha
```

### Dans `Up2C` — activation optionnelle

```python
class Up2C(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, attention=False):
        ...
        if attention:
            self.att = AttentionGate(F_g=F_g, F_x=out_channels, F_int=out_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)                      # x1 = gating signal g (vient du bas)
        if self.use_attention:
            x2 = self.att(g=x1, x=x2)         # x2 = skip connection filtrée
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
```

### Dans `UNet` — paramètre `attention`

```python
model = UNet(
    hidden_channels=[32, 64, 128, 256, 512],
    batch_norm=True,
    bilinear=True,
    attention=True    # active les 4 AG sur les 4 skip connections
)
```

---

## 7. Résumé visuel du filtrage

```
Skip connection brute (encodeur down1, 64×64) :

  ████████░░░░░░░░░░░░░░░░░░░░░░
  ████████░░░░░░░░░░░░░░░░░░░░░░   ← forte activation sur la faille
  ████████░░░░░░░░░░░░░░░░░░░░░░
  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ← activation faible sur le fond
  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Carte d'attention α (produite par AG) :

  0.92 0.90 0.88 0.05 0.03 0.04 0.06
  0.91 0.93 0.85 0.04 0.02 0.05 0.03
  0.89 0.91 0.87 0.06 0.03 0.04 0.05

Skip filtrée après x × α :

  ████████░░░░░░░  ← faille bien présente
  ████████░░░░░░░
  ████████░░░░░░░
  ░░░░░░░░░░░░░░░  ← fond quasi silencieux
  ░░░░░░░░░░░░░░░
```

---

## 8. Pourquoi ça marche mieux avec peu de données

Sans attention gate, le décodeur doit **apprendre seul** à ignorer les features de fond dans les skip connections. Avec 600 images seulement, il n'a pas assez d'exemples pour généraliser ce filtrage correctement.

Avec l'attention gate, le filtrage est **guidé structurellement** par l'architecture — le réseau n'a pas à l'apprendre from scratch, il apprend juste à pondérer. Résultat : convergence plus rapide, meilleure généralisation.

---

## 9. Référence

> Oktay et al., **"Attention U-Net: Learning Where to Look for the Pancreas"**, MIDL 2018.
> https://arxiv.org/abs/1804.03999

Contexte original : segmentation du pancréas en imagerie médicale — même problème que les failles géologiques (petit objet noyé dans un grand fond).