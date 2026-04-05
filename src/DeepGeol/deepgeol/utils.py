import os
import json
import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt

#=================================================
# Les constantes
#=================================================
PATH_TRAIN_DEMO = "/lium/buster1/larcher/M2/deep_learning/TP_CNN_UNet/data/training_data.npy"
PATH_TRAIN_MASKS_DEMO = "/lium/buster1/larcher/M2/deep_learning/TP_CNN_UNet/data/training_masks.npy"
PATH_TEST_DEMO = "/lium/buster1/larcher/M2/deep_learning/TP_CNN_UNet/data/test_data.npy"
PATH_TEST_MASKS_DEMO = "/lium/buster1/larcher/M2/deep_learning/TP_CNN_UNet/data/test_masks.npy"

SAVE_LOG_PATH = "log/"
SAVE_MODEL_PATH = os.path.join(SAVE_LOG_PATH, "best_model.pth")

FILE_NAME_FOR_LOG = os.path.basename(__file__)

# ================================================
# Fonctions pour les logs des runs
# ================================================

def get_model_summary(model):
    # Compter le nombre de paramètres du réseau
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Récupérer les types de blocs utilisés dans le réseau
    blocks = {}
    for name, module in model.named_children():
        blocks[name] = module.__class__.__name__

    # Rassembler les informations dans un dictionnaire pour ensuite
    # le mettre dans un json
    return {
        "class": model.__class__.__name__,
        "hidden_channels": model.hidden_channels,
        "input_channels": model.input_channels,
        "n_classes": model.n_classes,
        "bilinear": model.bilinear,
        "batch_norm": model.batch_norm,
        "dropout": model.dropout,
        "attention": model.attention,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "blocks": blocks,
    }


def get_environment_info():
    # Récupérer les informations sur l'environnement d'exécution
    env = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    # Si CUDA est dispo, ajouter les détails du GPU
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["cuda_version"] = torch.version.cuda
        env["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return env


def create_run_log(base_log_dir, model, config, results, training_loss, validation_loss, run_duration):
    # Créer un dossier de log avec la date et l'heure du run
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    run_dir = os.path.join(base_log_dir, date_str, time_str)
    os.makedirs(run_dir, exist_ok=True)

    # Un fichier json pour les infos du run et un fichier image
    # pour le plot de la courbe de loss
    json_path = os.path.join(run_dir, f"{time_str}.json")
    plot_path = os.path.join(run_dir, f"{time_str}_loss.png")

    # Création de section dans le json pour les infos du modèle, de la config,
    # des résultats et de l'environnement
    record = {
        "model": get_model_summary(model),
        "config": config,
        "results": results,
        "environment": get_environment_info(),
    }
    # On ajoute le temps de calcul
    results["training_time_seconds"] = run_duration

    # sauver le json et le plot de la courbe de loss
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    plt.figure()
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"[{FILE_NAME_FOR_LOG}] JSON saved at {json_path}")
    print(f"[{FILE_NAME_FOR_LOG}] Plot saved at {plot_path}")

    return run_dir

# ================================================
# Fonctions pour la soutenance
# ================================================
def load_data(path_data, path_masks):
    x = np.load(path_data)
    y = np.load(path_masks)
    return x, y

def visualize_image_and_mask(image, mask, idx=0):
    image = image[idx]
    mask = mask[idx]

    # (H, W,1) -> (H, W)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Associated Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()