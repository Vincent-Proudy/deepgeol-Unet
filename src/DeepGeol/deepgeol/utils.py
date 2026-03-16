import os
import json
import datetime
import torch
import matplotlib.pyplot as plt

FILE_NAME_FOR_LOG = os.path.basename(__file__)

def get_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # List all named blocks (one level deep — skips individual weights)
    blocks = {}
    for name, module in model.named_children():
        blocks[name] = module.__class__.__name__

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
    env = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["cuda_version"] = torch.version.cuda
        env["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return env


def create_run_log(base_log_dir, model, config, results, training_loss, validation_loss, run_duration):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    # Create dated + timestamped directory
    run_dir = os.path.join(base_log_dir, date_str, time_str)
    os.makedirs(run_dir, exist_ok=True)

    json_path = os.path.join(run_dir, f"{time_str}.json")
    plot_path = os.path.join(run_dir, f"{time_str}_loss.png")

    record = {
        "model": get_model_summary(model),
        "config": config,
        "results": results,
        "environment": get_environment_info(),
    }

    results["training_time_seconds"] = run_duration

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