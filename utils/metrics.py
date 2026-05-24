"""
OPTIMIZED Evaluation Metrics for Tooth Segmentation

Key improvements:
- Per-sample computation (accurate for batch_size > 1)
- Additional metrics: Precision, Recall, F1, Specificity, Accuracy, ROC-AUC
- Confusion matrix stats
- Boundary metrics (optional)
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


# =========================================================
# BASIC METRICS
# =========================================================

def dice_coefficient(pred, target, threshold=0.5, smooth=1e-5):
    """
    Dice Coefficient (F1 Score for binary segmentation)
    pred:   (B, num_classes, H, W)
    target: (B, H, W)
    """
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()

    batch_size = pred.shape[0]
    dice_scores = []

    for i in range(batch_size):
        pred_i = pred_binary[i]
        target_i = target_binary[i]

        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())

    return sum(dice_scores) / len(dice_scores)


def iou_score(pred, target, threshold=0.5, smooth=1e-5):
    """
    IoU (Jaccard Index)
    """
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()

    batch_size = pred.shape[0]
    iou_scores = []

    for i in range(batch_size):
        pred_i = pred_binary[i]
        target_i = target_binary[i]

        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum() - intersection

        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())

    return sum(iou_scores) / len(iou_scores)


# =========================================================
# ADDITIONAL METRICS
# =========================================================

def precision_score(pred, target, threshold=0.5, smooth=1e-5):
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()

    batch_size = pred.shape[0]
    scores = []

    for i in range(batch_size):
        TP = (pred_binary[i] * target_binary[i]).sum()
        FP = (pred_binary[i] * (1 - target_binary[i])).sum()

        precision = (TP + smooth) / (TP + FP + smooth)
        scores.append(precision.item())

    return sum(scores) / len(scores)


def recall_score(pred, target, threshold=0.5, smooth=1e-5):
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()

    batch_size = pred.shape[0]
    scores = []

    for i in range(batch_size):
        TP = (pred_binary[i] * target_binary[i]).sum()
        FN = ((1 - pred_binary[i]) * target_binary[i]).sum()

        recall = (TP + smooth) / (TP + FN + smooth)
        scores.append(recall.item())

    return sum(scores) / len(scores)


def specificity_score(pred, target, threshold=0.5, smooth=1e-5):
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()

    batch_size = pred.shape[0]
    scores = []

    for i in range(batch_size):
        TN = ((1 - pred_binary[i]) * (1 - target_binary[i])).sum()
        FP = (pred_binary[i] * (1 - target_binary[i])).sum()

        specificity = (TN + smooth) / (TN + FP + smooth)
        scores.append(specificity.item())

    return sum(scores) / len(scores)


def f1_score(pred, target, threshold=0.5, smooth=1e-5):
    precision = precision_score(pred, target, threshold, smooth)
    recall = recall_score(pred, target, threshold, smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    return f1


def accuracy(pred, target, threshold=0.5):
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()

    correct = (pred_binary == target_binary).float().sum()
    total = target_binary.numel()
    return (correct / total).item()


# =========================================================
# CONFUSION MATRIX
# =========================================================

def confusion_matrix_stats(pred, target, threshold=0.5):
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()

    TP = (pred_binary * target_binary).sum().item()
    TN = ((1 - pred_binary) * (1 - target_binary)).sum().item()
    FP = (pred_binary * (1 - target_binary)).sum().item()
    FN = ((1 - pred_binary) * target_binary).sum().item()

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Total": TP + TN + FP + FN
    }


# =========================================================
# COMPUTE ALL METRICS (FIXED PER-SAMPLE ROC-AUC)
# =========================================================

def compute_all_metrics(pred, target, threshold=0.5, smooth=1e-5):
    # Lấy xác suất lớp răng (Foreground) chuẩn xác sau Softmax
    probs = torch.softmax(pred, dim=1)[:, 1]
    pred_binary = (probs > threshold).float()
    target_binary = (target == 1).float()

    batch_size = pred.shape[0]

    # Khởi tạo các mảng tích lũy chỉ số
    accumulated_metrics = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "f1": [],
        "accuracy": [],
        "auc": []  # Đã chuyển auc vào mảng tích lũy per-sample
    }

    for i in range(batch_size):
        pred_i = pred_binary[i]
        target_i = target_binary[i]
        prob_i = probs[i]

        TP = (pred_i * target_i).sum()
        TN = ((1 - pred_i) * (1 - target_i)).sum()
        FP = (pred_i * (1 - target_i)).sum()
        FN = ((1 - pred_i) * target_i).sum()

        intersection = TP
        union = pred_i.sum() + target_i.sum()

        # Tính toán chi tiết từng mẫu
        dice = (2 * intersection + smooth) / (union + smooth)
        iou = (intersection + smooth) / (union - intersection + smooth)
        precision = (TP + smooth) / (TP + FP + smooth)
        recall = (TP + smooth) / (TP + FN + smooth)
        specificity = (TN + smooth) / (TN + FP + smooth)
        f1 = (2 * precision * recall) / (precision + recall + smooth)
        acc = (TP + TN) / (pred_i.numel() + smooth)

        accumulated_metrics["dice"].append(dice.item())
        accumulated_metrics["iou"].append(iou.item())
        accumulated_metrics["precision"].append(precision.item())
        accumulated_metrics["recall"].append(recall.item())
        accumulated_metrics["specificity"].append(specificity.item())
        accumulated_metrics["f1"].append(f1.item())
        accumulated_metrics["accuracy"].append(acc.item())

        # ===== TÍNH ROC-AUC RIÊNG CHO TỪNG ẢNH AN TOÀN =====
        y_true_i = target_i.contiguous().view(-1).cpu().numpy().astype(np.int32)
        y_prob_i = prob_i.contiguous().view(-1).cpu().numpy().astype(np.float32)

        # Chỉ tính toán nếu ảnh chứa đầy đủ cả 2 class (Nền + Răng)
        if len(np.unique(y_true_i)) == 2:
            try:
                auc_i = roc_auc_score(y_true_i, y_prob_i)
                accumulated_metrics["auc"].append(auc_i)
            except:
                pass  # Tránh gán mặc định gây lệch trung bình chéo

    # Tính toán kết quả trung bình cuối cùng một cách an toàn
    final_results = {}
    for k, v in accumulated_metrics.items():
        if len(v) > 0:
            final_results[k] = sum(v) / len(v)
        else:
            # Dự phòng nếu toàn bộ batch trống không tính được AUC
            final_results[k] = 0.5 if k == "auc" else 0.0

    return final_results


# =========================================================
# BOUNDARY METRIC
# =========================================================

def boundary_iou(pred, target, threshold=0.5, dilation=2):
    probs = torch.softmax(pred, dim=1)[:, 1]

    pred_binary = (probs > threshold).float().unsqueeze(1)
    target_binary = (target == 1).float().unsqueeze(1)

    kernel = torch.ones(1, 1, dilation * 2 + 1, dilation * 2 + 1).to(pred.device)

    pred_dilated = F.conv2d(pred_binary, kernel, padding=dilation) > 0
    pred_eroded = F.conv2d(pred_binary, kernel, padding=dilation) >= kernel.sum()

    pred_boundary = (pred_dilated.float() - pred_eroded.float()).squeeze(1)

    target_dilated = F.conv2d(target_binary, kernel, padding=dilation) > 0
    target_eroded = F.conv2d(target_binary, kernel, padding=dilation) >= kernel.sum()

    target_boundary = (target_dilated.float() - target_eroded.float()).squeeze(1)

    intersection = (pred_boundary * target_boundary).sum()
    union = pred_boundary.sum() + target_boundary.sum() - intersection

    return ((intersection + 1e-5) / (union + 1e-5)).item()


# =========================================================
# FIND BEST THRESHOLD
# =========================================================

def find_optimal_threshold(pred, target, metric="dice", search_range=(0.3, 0.7), steps=20):
    thresholds = torch.linspace(search_range[0], search_range[1], steps)

    metric_funcs = {
        "dice": dice_coefficient,
        "iou": iou_score,
        "f1": f1_score,
        "precision": precision_score,
        "recall": recall_score
    }

    metric_func = metric_funcs[metric]

    best_threshold = 0.5
    best_score = 0

    for thresh in thresholds:
        score = metric_func(pred, target, threshold=thresh.item())
        if score > best_score:
            best_score = score
            best_threshold = thresh.item()

    return best_threshold, best_score