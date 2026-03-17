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

from DeepGeol.deepgeol.unet import UNet
import DeepGeol.deepgeol.utils as utils
from DeepGeol.deepgeol.geoutil import GeoSet, get_crop_list

#================================================
# Constants for simplicty
#================================================

BATCH_SIZE = 64
SEUIL = 0.3  # Sigmoid threshold to binarize predictions
TRAIN_VAL_RATIO = 0.9  # Proportion of data used for training vs. validation

PATH_DATA_ORIGINAL  = "/projects/m26043/data/ArticDEM/"
PATH_MASK_ORIGINAL  = "/projects/m26043/data/masks/"
WINDOW_SIZE         = 256   # taille des patches extraits (doit correspondre à l'entrée du UNet)
STRIDE              = 128   # chevauchement entre patches (128 = 50% overlap)

FILE_NAME_FOR_LOG = os.path.basename(__file__)

#================================================
# preprocessing
#================================================
class GeoDataset(Dataset):
   """
   Custom Dataset for geological segmentation data.
   Images and masks are loaded from .npy files and preprocessed.

   """
   def __init__(self, data_path_x, data_path_masks):
       super(GeoDataset, self).__init__()

       # Load raw numpy arrays from disk
       self.x = np.load(data_path_x)
       self.y = np.load(data_path_masks)

       # Convert to float tensors and move directly to GPU
       # (dataset is small enough to fit entirely in VRAM)
       self.x = torch.from_numpy(self.x).float()
       self.y = torch.from_numpy(self.y).float()

       # numpy arrays are (N, H, W, C), PyTorch expects (N, C, H, W)
       self.x = self.x.permute(0, 3, 1, 2)
       self.y = self.y.permute(0, 3, 1, 2)

       # Convert RGB images to grayscale by averaging channels
       # The UNet takes a single-channel input
       self.x = self.x.mean(dim=1, keepdim=True)

   def __len__(self):
       return len(self.x)

   def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_data():
   """
   Loads and splits data into train, validation, and test DataLoaders.

   Returns:
       dataloader_train, dataloader_test, dataloader_val
   """
   dataset_train_full = GeoDataset(utils.PATH_TRAIN_DEMO, utils.PATH_TRAIN_MASKS_DEMO)
   dataset_test = GeoDataset(utils.PATH_TEST_DEMO, utils.PATH_TEST_MASKS_DEMO)

   # Split training data into train and validation sets
   train_set, val_set = torch.utils.data.random_split(
       dataset_train_full, [TRAIN_VAL_RATIO, 1 - TRAIN_VAL_RATIO]
   )

   dataloader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
   dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
   dataloader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

   return dataloader_train, dataloader_test, dataloader_val

def load_data_tif():
    """
    Loads training, validation and test DataLoaders from .tif files.
    Uses get_crop_list to enumerate all possible patches from the large images,
    then GeoSet to load each patch on the fly with rasterio (no full image in RAM).

    The 'safe_split' mode ensures train/val/test sets come from different spatial
    regions of each image — avoids data leakage due to overlapping patches.

    Returns:
        dataloader_train, dataloader_test, dataloader_val
    """
    # Build the list of all extractable patches per split
    # safe_split : each image is cut spatially so train/val/test don't overlap
    training_df, validation_df, dev_df = get_crop_list(
        data_path=PATH_DATA_ORIGINAL,
        mask_path=PATH_MASK_ORIGINAL,
        window_size=(WINDOW_SIZE, WINDOW_SIZE),
        stride=(STRIDE, STRIDE),
        train_val_dev=(0.8, 0.1, 0.1),
        splitting_mode='safe_split',
        crop_mode='vertical'
    )

    # Reset index so GeoSet can access rows by integer index
    training_df = training_df.reset_index(drop=True)
    validation_df = validation_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)

    dataset_train = GeoSet(training_df, PATH_DATA_ORIGINAL, PATH_MASK_ORIGINAL, WINDOW_SIZE)
    dataset_val = GeoSet(validation_df, PATH_DATA_ORIGINAL, PATH_MASK_ORIGINAL, WINDOW_SIZE)
    dataset_test = GeoSet(dev_df, PATH_DATA_ORIGINAL, PATH_MASK_ORIGINAL, WINDOW_SIZE)

    # num_workers > 0 is now safe: data is loaded from disk on CPU, not stored on GPU
    # pin_memory=True speeds up the CPU→GPU transfer in the training loop
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)
    dataloader_val = DataLoader(dataset_val,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(dataset_test,  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=4, pin_memory=True)

    return dataloader_train, dataloader_test, dataloader_val

#================================================
# Main functions for training and evaluation
#================================================
def train(dataloader, model, loss_fn, optimizer, scaler, device):
    """
    Runs one full training epoch over the dataloader.

    Uses Automatic Mixed Precision (AMP) to speed up training on GPU:
    forward pass is computed in float16, reducing memory usage and
    increasing throughput, while the GradScaler prevents underflow
    in the backward pass.

    The scaler is passed as a parameter so its internal state
    persists correctly across epochs.

    Returns:
        mean training loss over all batches
    """
    model.train()
    losses = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # AMP: compute forward pass in float16 for speed
        with autocast('cuda'):
            output = model(x)
            loss_value = loss_fn(output, y)

        # Scale gradients to avoid float16 underflow, then backpropagate
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss_value.item())

    return np.mean(losses)


def evaluate(dataloader, model, loss_fn, device):
    """
    Evaluates the model on a given dataloader (validation or test set).

    Computes pixel-wise accuracy, mean loss, Dice F1 score, and jaccard_index.
    All intermediate computations are kept on GPU to avoid costly
    CPU transfers at each batch; results are only moved to CPU at the end.

    Returns:
        accuracy (%), avg_loss, f1, jaccard_index
    """
    model.eval()
    total_loss, num_batches, num_pixels = 0.0, 0, 0

    # Accumulate TP/FP/FN entirely on GPU
    total_tp = torch.tensor(0, device=device)
    total_fp = torch.tensor(0, device=device)
    total_fn = torch.tensor(0, device=device)
    total_correct = torch.tensor(0, device=device)

    # Assure no gradient calculating during evaluation
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss_value = loss_fn(pred, y)
            total_loss += loss_value.item()
            num_batches += 1

            # Apply sigmoid then threshold to obtain binary mask
            pred_probs = torch.sigmoid(pred)
            pred_mask = (pred_probs > SEUIL).long()
            y_true = y.long()

            total_tp += torch.logical_and(pred_mask == 1, y_true == 1).sum()
            total_fp += torch.logical_and(pred_mask == 1, y_true == 0).sum()
            total_fn += torch.logical_and(pred_mask == 0, y_true == 1).sum()
            total_correct += (pred_mask == y_true).sum()
            num_pixels += torch.numel(pred_mask)

    # Compute global metrics...
    # Move results to CPU only once, at the end of the loop
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = (total_correct.item() / num_pixels) * 100 if num_pixels > 0 else 0.0

    tp = total_tp.item()
    fp = total_fp.item()
    fn = total_fn.item()

    # F1: mean of precision and recall
    denom = 2 * tp + fp + fn
    f1 = (2 * tp) / denom if denom > 0 else 0.0

    # jaccard_index (IoU / Intersection over Union): standard metric
    # for segmentation tasks.More informative than pixel accuracy
    # on imbalanced datasets
    iou_denom = tp + fp + fn
    jaccard_index = tp / iou_denom if iou_denom > 0 else 0.0

    return accuracy, avg_loss, f1, jaccard_index


def train_loop(epochs=10, device=None):
    """
    Full training pipeline: data loading, model init, training loop,
    early stopping, curve visualization, and logging.

    Args:
        epochs: maximum number of training epochs
        device: torch device string ('cuda' or 'cpu'). Auto-detected if None.

    Returns:
        trained model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{FILE_NAME_FOR_LOG}] Using {'GPU' if device == 'cuda' else 'CPU'} device")

    # Number of epochs without improvement before stopping training
    patience = 8

    dataloader_train, _, dataloader_val = load_data()

    # Smaller hidden_channels than the original UNet default [64,128,256,512,1024]
    # With only 600 training images, a lighter model reduces overfitting risk
    # and is much faster to train (fewer params to compute)
    model = UNet(
        hidden_channels=[32, 64, 128, 256, 512],
        batch_norm=True,
        dropout=False,
        bilinear=False,
        attention=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=5)

    # pos_weight penalizes false negatives more heavily to handle class imbalance:
    # geological structures occupy a small fraction of each image
    pos_weight = torch.tensor([4.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # GradScaler must be created once and reused across epochs so that its
    # internal loss scale factor can adapt progressively during training
    scaler = GradScaler('cuda')

    training_loss = []
    validation_loss = []
    best_metrics = {"loss": float("inf"), "acc": 0.0, "f1": 0.0, "jaccard_index": 0.0}
    patience_count = 0
    early_stopping = False
    early_stopping_value = -1

    start_time = time.time()

    for e in tqdm.tqdm(range(epochs), desc="Training Progress", unit="epoch"):

        avg_train_loss = train(dataloader_train, model, loss_fn, optimizer, scaler, device)
        training_loss.append(avg_train_loss)

        acc, avg_val_loss, f1, jaccard_index = evaluate(dataloader_val, model, loss_fn, device)
        validation_loss.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step()

        # Save the model whenever validation loss improves (early stopping)
        if avg_val_loss < best_metrics["loss"]:
            best_metrics = {"loss": avg_val_loss, "acc": acc, "f1": f1, "jaccard_index": jaccard_index}
            patience_count = 0
            os.makedirs(utils.SAVE_LOG_PATH, exist_ok=True)
            torch.save(model.state_dict(), utils.SAVE_MODEL_PATH)
        else:
            patience_count += 1

        if patience_count >= patience:
            early_stopping = True
            early_stopping_value = e+1
            print(f"[{FILE_NAME_FOR_LOG}] Early stopping triggered at epoch {e + 1}")
            break

    end_time = time.time()

    # reload the best weights before returning
    model.load_state_dict(torch.load(utils.SAVE_MODEL_PATH))

    # Plot training vs validation loss curves to visualize convergence
    plt.figure()
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

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

    temp_path = os.path.join(run_dir, "final_model.pth")
    torch.save(model.state_dict(), temp_path)
    os.remove(utils.SAVE_MODEL_PATH)
    print(f"[{FILE_NAME_FOR_LOG}] Final model saved at {temp_path}")

    return model


def test(model, device=None):
    """
    Evaluates the model on the held-out test set and prints final metrics.

    Args:
        model: trained UNet model
        device: the device to run the evaluation

    Returns:
        accuracy (%), test_loss, f1, jaccard_index
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{FILE_NAME_FOR_LOG}] Using {'GPU' if device == 'cuda' else 'CPU'} device")

    _, dataloader_test, _ = load_data()

    # Use the same loss (with pos_weight) as during training for consistency
    pos_weight = torch.tensor([4.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    accuracy, test_loss, f1, jaccard_index = evaluate(dataloader_test, model, loss_fn, device)

    print(
        f"[{FILE_NAME_FOR_LOG}] Test metrics : \n"
        f"  Accuracy: {accuracy:.2f}%\n"
        f"  Loss: {test_loss:.6f}\n"
        f"  F1: {f1:.4f}\n"
        f"  Jaccard Index: {jaccard_index:.4f}"
    )
    return accuracy, test_loss, f1, jaccard_index

#================================================
# Main execution
#================================================
if __name__ == "__main__":
    # print(f"[{FILE_NAME_FOR_LOG}] Starting training loop...")
    # trained_model = train_loop(epochs=50)
    # accuracy, loss, f1, jaccard_index = test(trained_model)
    training_df, validation_df, dev_df = get_crop_list(
        data_path=PATH_DATA_ORIGINAL,
        mask_path=PATH_MASK_ORIGINAL,
        window_size=(256, 256),
        stride=(128, 128),
        train_val_dev=(0.8, 0.1, 0.1),
        splitting_mode='safe_split',
    )
    print(training_df.head())
    print(f"Train: {len(training_df)}, Val: {len(validation_df)}, Test: {len(dev_df)}")
