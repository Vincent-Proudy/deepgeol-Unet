# coding: utf-8 -*-

# For this exercise we have access to a small amount of data (the domain being recent
# and the data manually annotated, there are few samples).
#
# The data is accessible in /lium/buster1/larcher/M2/deep_learning/TP_CNN_UNet/data/.
#
# The directory contains a file with 600 training images
# The directory contains a file with the 600 corresponding masks
# The directory also contains a file with 70 test images
# Finally a file with the 70 test masks is available in the same directory

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torch.amp import GradScaler, autocast
import os
import time
import sys

sys.path.append("../../DeepGeol/")
from DeepGeol.deepgeol.unet import UNet
import DeepGeol.deepgeol.utils as utils
# from DeepGeol.deepgeol.geolutils import GeoSet, get_crop_list

#============================================================================
# Les constantes et hyperparamètres sont
# centralisés ici pour faciliter les modifications et
# éviter les valeurs dispersés dans le code.
#============================================================================
# PATH_DATA_ORIGINAL  = "/projects/m26043/data/ArticDEM/"
# PATH_MASK_ORIGINAL  = "/projects/m26043/data/masks/"
# WINDOW_SIZE = 256   # taille des patches extraits (doit correspondre à l'entrée du UNet)
# STRIDE = 128   # chevauchement entre patches (128 = 50% overlap)

BATCH_SIZE = 64
LR = 1e-4
# Le seuil de la signoide lors des prédictions pour convertir les probabilités en masques binaires.
SEUIL = 0.3
# Quand je split, c'est le ratio entre le train et le val
TRAIN_VAL_RATIO = 0.9
# Nom du fichier pour l'affichage des print (savoir d'où ils viennent)
FILE_NAME_FOR_LOG = os.path.basename(__file__)

#============================================================================
# preprocessing des données
#============================================================================
class GeoDataset(Dataset):
    """
    Classe pour le chargement les images de géologies à partir de fichiers .npy
    """
    def __init__(self, data_path_x, data_path_masks):
        super(GeoDataset, self).__init__()

        # On met les données dans des numpy array
        self.x = np.load(data_path_x)
        self.y = np.load(data_path_masks)

        # Convertir les numpy arrays en tensors PyTorch et les mettre au bon format pour le UNet
        self.x = torch.from_numpy(self.x).float()
        self.y = torch.from_numpy(self.y).float()

        # Les données sont au format (N, H, W, C) et le UNet attend
        # (N, C, H, W), on permute les dimensions
        self.x = self.x.permute(0, 3, 1, 2)
        self.y = self.y.permute(0, 3, 1, 2)

        # les données d'entrée ont 3 canaux (RGB), mais le UNet doit prendre un seul canal en entrée.
        # Donc on mets des nuances de gris en moyennant le RGB
        self.x = self.x.mean(dim=1, keepdim=True)

    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_data(batch_size=BATCH_SIZE):
    """
    Fonction qui est responsable de charger les données d'entraînement, de validation et de test
    à partir des fichiers .npy.
    """
    dataset_train_full = GeoDataset(utils.PATH_TRAIN_DEMO, utils.PATH_TRAIN_MASKS_DEMO)
    dataset_test = GeoDataset(utils.PATH_TEST_DEMO, utils.PATH_TEST_MASKS_DEMO)

    # On divise le train original en train et val
    train_set, val_set = torch.utils.data.random_split(
        dataset_train_full, [TRAIN_VAL_RATIO, 1 - TRAIN_VAL_RATIO]
    )

    # Création des 3 dataloaders pour le train, le test et la validation
    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    dataloader_val = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return dataloader_train, dataloader_test, dataloader_val

# def load_data_tif():
#     training_df, validation_df, dev_df = get_crop_list(
#         data_path=PATH_DATA_ORIGINAL,
#         mask_path=PATH_MASK_ORIGINAL,
#         window_size=(WINDOW_SIZE, WINDOW_SIZE),
#         stride=(STRIDE, STRIDE),
#         train_val_dev=(0.8, 0.1, 0.1),
#         splitting_mode='safe_split',
#         crop_mode='vertical'
#     )
#
#     training_df = training_df.reset_index(drop=True)
#     validation_df = validation_df.reset_index(drop=True)
#     dev_df = dev_df.reset_index(drop=True)
#
#     dataset_train = GeoSet(training_df, PATH_DATA_ORIGINAL, PATH_MASK_ORIGINAL, WINDOW_SIZE)
#     dataset_val = GeoSet(validation_df, PATH_DATA_ORIGINAL, PATH_MASK_ORIGINAL, WINDOW_SIZE)
#     dataset_test = GeoSet(dev_df, PATH_DATA_ORIGINAL, PATH_MASK_ORIGINAL, WINDOW_SIZE)
#
#     dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,
#                                   num_workers=4, pin_memory=True)
#     dataloader_val = DataLoader(dataset_val,   batch_size=BATCH_SIZE, shuffle=False,
#                                   num_workers=4, pin_memory=True)
#     dataloader_test = DataLoader(dataset_test,  batch_size=BATCH_SIZE, shuffle=False,
#                                   num_workers=4, pin_memory=True)
#
#     return dataloader_train, dataloader_test, dataloader_val

#============================================================================
# Fonction pour le training loop et l'évaluation du modèle
#============================================================================
def train(dataloader, model, loss_fn, optimizer, scaler, device):
    model.train()
    losses = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # On fait de la précision mixte pour accélérer les calculs
        with autocast('cuda'):
            # on peut faire du float16 pour les entrées et les poids du modèle, mais on garde les
            # calculs de la loss en float32 pour éviter les problèmes de précision
            output = model(x)
            loss_value = loss_fn(output, y)

        # On va devoir faire du gradient scaling pour permet éviter les problèmes de gradients
        # qui deviennent trop petits lors de l'utilisation du float16
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss_value.item())

    return np.mean(losses)


def evaluate(dataloader, model, loss_fn, device):
    model.eval()

    # Variables pour accumuler les métriques
    total_loss, num_batches, num_pixels = 0.0, 0, 0

    # on met sur le GPU pour éviter les problèmes de mémoire lors du calcul
    # des métriques sur tous les pixels
    total_tp = torch.tensor(0, device=device)
    total_fp = torch.tensor(0, device=device)
    total_fn = torch.tensor(0, device=device)
    total_correct = torch.tensor(0, device=device)

    # Fait attention qu'on ne calcul pas de gradients pendant la phase
    # d'évaluation du réseau de neuronnes
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss_value = loss_fn(pred, y)
            total_loss += loss_value.item()
            num_batches += 1

            # On va appliquer une signoide pour passer de probabilités à des prédictions binaires,
            # j'ai mis un seuil de 0.3 qui est plus que tolérant (peut être trop ?)
            pred_probs = torch.sigmoid(pred)
            pred_mask = (pred_probs > SEUIL).long()
            y_true = y.long()

            # Calcul des métriques de classification binaire pour les masques
            total_tp += torch.logical_and(pred_mask == 1, y_true == 1).sum()
            total_fp += torch.logical_and(pred_mask == 1, y_true == 0).sum()
            total_fn += torch.logical_and(pred_mask == 0, y_true == 1).sum()
            total_correct += (pred_mask == y_true).sum()
            # compte le nombre total de pixels pour l'accuracy
            num_pixels += torch.numel(pred_mask)

    # loss moyenne sur tous les batches
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    # Accuracy pixel par pixel
    accuracy = (total_correct.item() / num_pixels) * 100 if num_pixels > 0 else 0.0

    # on remet sur le CPU pour éviter une erreur
    tp = total_tp.item()
    fp = total_fp.item()
    fn = total_fn.item()

    # Calcul du F1-score
    denom = 2 * tp + fp + fn
    f1 = (2 * tp) / denom if denom > 0 else 0.0

    # Calcul de l'index de Jacard, pertinent dans notre cas pour le recouvrement des masks
    iou_denom = tp + fp + fn
    jaccard_index = tp / iou_denom if iou_denom > 0 else 0.0

    return accuracy, avg_loss, f1, jaccard_index


def train_loop(epochs=10, device=None, batch_size=BATCH_SIZE, lr=LR):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{FILE_NAME_FOR_LOG}] Using {'GPU' if device == 'cuda' else 'CPU'} device")

    # Nombre d'épochs où la loss de validation ne s'améliore pas afin d'éviter
    # l'overfitting du réseau = early stopping
    patience = 8

    dataloader_train, _, dataloader_val = load_data(batch_size=batch_size)

    model = UNet(
        hidden_channels=[32, 64, 128, 256, 512],
        batch_norm=True,
        dropout=False,
        bilinear=False,
        attention=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=5)

    # Comme notre dataset est très déséquilibré (beaucoup plus de pixels négatifs que positifs), on va
    # utiliser une loss avec un pos_weight, qui va en quelque sorte dire, une erreur sur un pixel positif est 4
    # fois plus grave qu'une erreur sur un pixel négatif
    pos_weight = torch.tensor([4.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # On déclare le scaler dans la boucle principale pour pas qu'il soit réinitialisé à chaque epoch, ce qui permet
    # de mieux gérer les problèmes de gradients lorsqu'on utilise du float16
    scaler = GradScaler('cuda')

    # Variables pour le suivi de la loss et des métriques, ainsi que pour l'early stopping
    training_loss = []
    validation_loss = []
    best_metrics = {"loss": float("inf"), "acc": 0.0, "f1": 0.0, "jaccard_index": 0.0}
    patience_count = 0
    early_stopping = False
    early_stopping_value = -1

    start_time = time.time()

    for e in tqdm.tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        # Entrainement + récupération de la loss moyenne sur le train
        avg_train_loss = train(dataloader_train, model, loss_fn, optimizer, scaler, device)
        training_loss.append(avg_train_loss)

        # Evaluation + récupération des métriques sur le set de validation
        acc, avg_val_loss, f1, jaccard_index = evaluate(dataloader_val, model, loss_fn, device)
        validation_loss.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step()

        # Si la loss de validation s'améliore, on sauvegarde et on remet la patience a 0
        # car il continu de converger
        if avg_val_loss < best_metrics["loss"]:
            best_metrics = {"loss": avg_val_loss, "acc": acc, "f1": f1, "jaccard_index": jaccard_index}
            patience_count = 0
            os.makedirs(utils.SAVE_LOG_PATH, exist_ok=True)
            torch.save(model.state_dict(), utils.SAVE_MODEL_PATH)
        else:
            # Réseau ne converge plus
            patience_count += 1

        # Quand ca fait trop longtemps que le réseau ne converge plus, alors on va arrêter
        # l'entraînement pour l'overfitting
        if patience_count >= patience:
            early_stopping = True
            early_stopping_value = e+1
            print(f"[{FILE_NAME_FOR_LOG}] Early stopping triggered at epoch {e + 1}")
            break

    end_time = time.time()

    # Avant de renvoyer le modèle final, on charge les poids du meilleur modèle sauvegardé pendant l'entraînement
    # car le dernier modèle n'est pas forcément le meilleur
    model.load_state_dict(torch.load(utils.SAVE_MODEL_PATH))

    # Visualisation de la courbe de loss pour le train et la validation
    plt.figure()
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Création d'un log pour la run, avec les différentes infos utiles pour la reproductibilité,
    # le suivi des expériences et les résultats obtenus
    run_dir = utils.create_run_log(
        base_log_dir=utils.SAVE_LOG_PATH,
        model=model,
        config={
            "Model": model.__class__.__name__,
            "Hidden channels": model.hidden_channels,
            "Bilinear": model.bilinear,
            "Batch norm": model.batch_norm,
            "Dropout": model.dropout,
            "Attention": model.attention,
            "Sigmoid threshold": SEUIL,
            "Epochs": epochs,
            "Batch size": BATCH_SIZE,
            "Train ratio": TRAIN_VAL_RATIO,
            "Optimizer": optimizer.__class__.__name__,
            "Scheduler": scheduler.__class__.__name__ if scheduler is not None else "None",
            "Learning rate": optimizer.param_groups[0]["lr"],
            "Loss function": loss_fn.__class__.__name__,
            "pos_weight": pos_weight.item(),
            "Patience": patience,
        },
        results={
            "Early stopping ": early_stopping,
            "Early stopping epoch": early_stopping_value,
            "Best validation loss": best_metrics["loss"],
            "Best validation acc": best_metrics["acc"],
            "Best validation F1": best_metrics["f1"],
            "Best validation Jaccard Index": best_metrics["jaccard_index"],
        },
        training_loss=training_loss,
        validation_loss=validation_loss,
        run_duration=(end_time - start_time),
    )

    # On supprime le fichier temporaire du meilleur modèle pendant l'entraînement pour ne garder que le modèle final
    # dans les logs, et éviter d'avoir des fichiers en trop
    temp_path = os.path.join(run_dir, "final_model.pth")
    torch.save(model.state_dict(), temp_path)
    os.remove(utils.SAVE_MODEL_PATH)
    print(f"[{FILE_NAME_FOR_LOG}] Final model saved at {temp_path}")

    return model


def test(model, device=None, batch_size=BATCH_SIZE):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{FILE_NAME_FOR_LOG}] Using {'GPU' if device == 'cuda' else 'CPU'} device")

    _, dataloader_test, _ = load_data(batch_size=batch_size)

    # On utilise la même loss que pour l'entraînement, avec le même pos_weight pour être
    # cohérent dans l'évaluation du modèle
    pos_weight = torch.tensor([4.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Evaluation du modèle sur le set de test
    accuracy, test_loss, f1, jaccard_index = evaluate(dataloader_test, model, loss_fn, device)

    print(
        f"[{FILE_NAME_FOR_LOG}] Test metrics : \n"
        f"  Accuracy: {accuracy:.2f}%\n"
        f"  Loss: {test_loss:.6f}\n"
        f"  F1: {f1:.4f}\n"
        f"  Jaccard Index: {jaccard_index:.4f}"
    )
    return accuracy, test_loss, f1, jaccard_index

#============================================================================
# Le point d'entrée du script, qui lance la boucle d'entraînement
# et affiche les résultats de test.
#============================================================================
if __name__ == "__main__":

    print(f"[{FILE_NAME_FOR_LOG}] Starting training loop...")
    trained_model = train_loop(epochs=50)
    accuracy, loss, f1, jaccard_index = test(trained_model)

    # training_df, validation_df, dev_df = get_crop_list(
    #     data_path=PATH_DATA_ORIGINAL,
    #     mask_path=PATH_MASK_ORIGINAL,
    #     window_size=(256, 256),
    #     stride=(128, 128),
    #     train_val_dev=(0.8, 0.1, 0.1),
    #     splitting_mode='safe_split',
    # )
    # print(training_df.head())
    # print(f"Train: {len(training_df)}, Val: {len(validation_df)}, Test: {len(dev_df)}")
