"""
OPTIMIZED Training script - Mamba-UNet

Features
- Learning rate warmup
- Gradient clipping
- Early stopping
- Model checkpointing
- Weighted sampling for broken teeth
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from models.mamba_unet import create_mamba_unet
from datasets.tooth_dataset import ToothDataset
from utils.losses import get_loss
from utils.metrics import compute_all_metrics, dice_coefficient, iou_score


# ======================================================
# TRAIN ONE EPOCH
# ======================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device,
                epoch, warmup_epochs=5, base_lr=0.0001):

    model.train()

    total_loss = 0
    total_dice = 0
    total_iou = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} - Training')

    for batch_idx, (images, masks) in enumerate(pbar):

        images = images.to(device)
        masks = masks.to(device)

        # ===== Warmup =====
        if epoch <= warmup_epochs:

            warmup_factor = (epoch - 1 + batch_idx / len(loader)) / warmup_epochs
            lr = base_lr * warmup_factor

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        with autocast():

            outputs = model(images)
            loss = criterion(outputs, masks)

        optimizer.zero_grad()

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():

            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)

        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{dice:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    n = len(loader)

    return (
        total_loss / n,
        total_dice / n,
        total_iou / n
    )


# ======================================================
# VALIDATION
# ======================================================

@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    metrics_sum = None

    for images, masks in tqdm(loader, desc="Validation"):

        images = images.to(device)
        masks = masks.to(device)

        with autocast():

            outputs = model(images)
            loss = criterion(outputs, masks)

        total_loss += loss.item()

        metrics = compute_all_metrics(outputs, masks)

        if metrics_sum is None:
            metrics_sum = {k: metrics[k] for k in metrics}
        else:
            for k in metrics:
                metrics_sum[k] += metrics[k]

    n = len(loader)

    avg_metrics = {k: v / n for k, v in metrics_sum.items()}

    return total_loss / n, avg_metrics


# ======================================================
# MAIN
# ======================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./data/d2')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=40)

    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--depths', type=int, nargs='+', default=[2,2,2,1])

    args = parser.parse_args()


    # ===== Setup =====

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(
        args.save_dir,
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(42)
        cudnn.benchmark = True
        cudnn.deterministic = False


    print("=" * 80)
    print(" MAMBA-UNET OPTIMIZED TRAINING")
    print(f"Device: {device}")
    print("=" * 80)


    # ======================================================
    # DATASET
    # ======================================================

    train_ds = ToothDataset(args.data_path, "train", args.img_size, augment=True)
    val_ds   = ToothDataset(args.data_path, "val", args.img_size, augment=False)


    # ===== Weighted Sampler =====

    if hasattr(train_ds, "sample_weights"):

        sampler = WeightedRandomSampler(
            weights=train_ds.sample_weights,
            num_samples=len(train_ds),
            replacement=True
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        n_broken = sum(1 for w in train_ds.sample_weights if w > 1)

        print(f"Broken samples in train: {n_broken}")

    else:

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )


    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")


    # ======================================================
    # MODEL
    # ======================================================

    model = create_mamba_unet(
        in_chans=1,
        num_classes=2,
        img_size=args.img_size,
        depths=args.depths,
        embed_dim=args.embed_dim
    ).to(device)


    print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


    # ======================================================
    # OPTIMIZER
    # ======================================================

    criterion = get_loss(version="improved")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.lr * 0.01
    )


    scaler = GradScaler()

    best_dice = 0
    patience_counter = 0


    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }


    # ======================================================
    # TRAIN LOOP
    # ======================================================

    for epoch in range(1, args.epochs + 1):

        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_dice, train_iou = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch,
            args.warmup_epochs,
            args.lr
        )

        val_loss, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device
        )

        val_dice = val_metrics["dice"]
        val_iou  = val_metrics["iou"]


        if epoch > args.warmup_epochs:
            scheduler.step()


        print(f"Val Dice: {val_dice:.4f}")


        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)

        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)


        # ===== Save best =====

        if val_dice > best_dice:

            best_dice = val_dice
            patience_counter = 0

            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best.pth")
            )

            print("✅ BEST MODEL SAVED")

        else:

            patience_counter += 1

            if patience_counter >= args.early_stop_patience:

                print("🛑 EARLY STOPPING")
                break


    # ======================================================
    # SAVE TRAINING CURVES
    # ======================================================

    plot_path = os.path.join(save_dir, "training_curves_full.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    epochs_range = range(1, len(history["train_loss"]) + 1)


    # Loss
    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs_range, history["val_loss"], label="Val Loss")

    axes[0].set_title("Loss over Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)


    # Dice
    axes[1].plot(epochs_range, history["train_dice"], label="Train Dice")
    axes[1].plot(epochs_range, history["val_dice"], label="Val Dice")

    axes[1].set_title("Dice over Epochs")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(True)


    # IoU
    axes[2].plot(epochs_range, history["train_iou"], label="Train IoU")
    axes[2].plot(epochs_range, history["val_iou"], label="Val IoU")

    axes[2].set_title("IoU over Epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].legend()
    axes[2].grid(True)


    fig.suptitle(
        f"Training Summary - Best Val Dice: {best_dice:.4f}",
        fontsize=16
    )

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


    print(f"\n📊 Training curves saved to: {plot_path}")
    print("Training completed.")
    print(f"Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()