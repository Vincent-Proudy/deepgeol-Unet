<br>
<div style="border: solid 3px #000;">
    <h1 style="text-align: center; color:#000; font-family:Georgia; font-size:26px;">Apprentissage profond, réseaux convolutionnels</h1>
    <p style='text-align: center;'>Master Informatique</p>
    <p style='text-align: center;'> Enseignant Chercheur : <strong> Anhony Larcher</strong> </p>
</div>


L'objectif de ce travail est d'implémenter un réseau convolutionnel pour la segmentation sémantique d'image. \
Cette tâche consiste à classifier chaque pixel de l'image de départ. \
Dans notre application, il s'agit d'une classification binaire, on parle de **masque** mais il serait possible de considérer plus de classes.

---
**Contexte**

Les géologues ont besoin de détecter et caractériser les **sub-glacial bedforms**, des sortes de *dunes de sédiments* qui se forment lors du déplacement de glaciers.\
Afin de mmieux comprendre le déplacement des glaciers ou des calottes polaires en période de fonte accélérée, ils anaysent les images satelittes et notamment les images topographiques afin de détecter les reliefs correspondant à ces formations.

L'apprentissage profond est utilisé depuis peu dans ce domaine et deux exemples d'images topographiques avec les **masques** produits sont donnés en exemple ci-dessous.

<center>
<img src='https://git-lium.univ-lemans.fr/cours_m2/open_resources/-/raw/master/figures/segmentation_semantique.png?ref_type=heads' width="500" height="500" >
</center>

---

## La tâche et les données

Pour cet exercice vous aurez accès à une petite quantité de données (le domaine étant récent et les données annotées manuellement, elles sont peu nombreuses).\
Les données sont acessibles dans `/lium/buster1/larcher/M2/deep_learning/TP_CNN_UNet/data/`.

- le répertoire contient un fichier correspondant à 600 images d'entraînement
- le répertoire contient un fichier correspondant aux 600 masques corespondant
- le répertoire également une fichier avec 70 images de tests
- enfin un fichier avec les 70 masques de tests est disponible dans le même réportoire

Pour cet exercice vous allez implémenter un réseau de type `U-Net`. \
Il en existe de nombreuses variantes pour des tâches différentes. \
Ce réseau est constitué de couches convolutionnelles à 2 dimensions organisées en blocs de trois types:

- **DoubleConv** qui inclue 2 couches convolutionnelles suivies chacune d'une `BatchNorm2d` (qui n'est pas dessinée sur la figure). Le bloc **DoubleConv** doit être modulaire (tailles de kernels, stride, padding et dilation paramétrables ainsi que la possibilité d'ajouter une couche de **DropOut** juste avant la sortie.
- **Down** qui incluent un sous échantillonage que vous implémenterez d'abord sous la forme d'un MaxPoling2d et un bloc **DoubleConv**, 
- **Up** qui incluent un sur-échantillonage que vous implémenterez d'abord avec une couche **ConvTranspose2d** et un bloc **DoubleConv**. le nombre de canaux intermédiaire dans le bloc `DoubleConv` est égal à la moitié des canaux d'entrée.

<img src='https://git-lium.univ-lemans.fr/cours_m2/open_resources/-/raw/master/figures/UNet.png?ref_type=heads' width="1000" height="200" >


Ces trois blocs doivent être implémentés de la façon. la plus flexible possible.

<div class="warning" style='background-color:#E9D8FD; color: #69337A; border-left: solid #805AD5 4px; border-radius: 4px; padding:0.7em;'>
<span>
<p style='margin-top:1em; text-align:center'>
<b>Contraintes d'implémentation</b></p>
<p style='margin-left:1em;'>
La classe implémentée pour le réseau <strong>U-Net</strong> devra être écrite sous forme d'un module Python (pas dans un notebook) <br>
afin d'être importée et exécutée dans un script <strong>SBATCH</strong> pour entrainer ce réseau sur le cluster du LIUM.<br>
Le framework vous est donné dans le répertoire <strong>/lium/buster1/larcher/M2/deep_learning/TP_CNN_UNet/DeepGeol/</strong>.<br>
L'API du module est donnée dans les fichiers <strong>*.py</strong> de ce répertoire. 

</p>
<p style='margin-bottom:1em; margin-right:1em; text-align:right; font-family:Georgia'>
</p></span>
</div>

---

## Améliorez votre système en implémentant un mécanisme d'attention

De nombreux mécanismes d'attention peuvent être ajoutés à un U-Net, essayez d'implémenter un des mécanismes présent dans la litérature.

---

## Ajouter un module ASPP

<img src='https://git-lium.univ-lemans.fr/cours_m2/open_resources/-/raw/master/figures/ASPP.png?ref_type=heads' width="1000" height="400" >
