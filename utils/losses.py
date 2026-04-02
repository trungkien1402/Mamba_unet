"""
OPTIMIZED Losses for Tooth Segmentation
3 versions: BASIC (current) | IMPROVED | ADVANCED
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# VERSION 1: BASIC (Your current version - WORKS FINE)
# ============================================================================

class DiceLoss_Basic(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        probs_fg = probs[:, 1, :, :]
        targets_fg = (targets == 1).float()

        dims = (1, 2)
        intersection = (probs_fg * targets_fg).sum(dims)
        union = probs_fg.sum(dims) + targets_fg.sum(dims)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice = dice.mean()

        return 1 - dice


class CombinedLoss_Basic(nn.Module):
    """Version hiện tại của bạn - ĐỦ TỐT để bắt đầu"""
    def __init__(self, weight_ce=0.3, weight_dice=0.7):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss_Basic()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
    
    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.weight_ce * ce + self.weight_dice * dice


# ============================================================================
# VERSION 2: IMPROVED (Recommended - Tối ưu cho răng)
# ============================================================================

class DiceLoss(nn.Module):
    """
    Cải thiện:
    - Tính cho CẢ 2 classes (background + foreground)
    - Weighted average
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        logits: (B, num_classes, H, W)
        targets: (B, H, W)
        """
        probs = F.softmax(logits, dim=1)  # (B, 2, H, W)
        num_classes = logits.shape[1]
        
        # Convert targets to one-hot
        targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        dice_per_class = []
        for c in range(num_classes):
            pred_c = probs[:, c]
            target_c = targets_onehot[:, c]
            
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            
            dice_c = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice_c.mean())
        
        # Weighted average - class răng quan trọng hơn
        dice_bg = dice_per_class[0]
        dice_fg = dice_per_class[1]
        
        dice_avg = 0.3 * dice_bg + 0.7 * dice_fg  # Teeth more important
        
        return 1 - dice_avg


class FocalLoss(nn.Module):
    """
    Focal Loss - Focus vào hard examples
    Tốt cho class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        logits: (B, num_classes, H, W)
        targets: (B, H, W)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss_Improved(nn.Module):
    """
    RECOMMENDED: CE + Dice + Focal
    Tối ưu cho segmentation răng
    """
    def __init__(self, weight_ce=0.2, weight_dice=0.5, weight_focal=0.3):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
    
    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        
        total_loss = (self.weight_ce * ce + 
                     self.weight_dice * dice + 
                     self.weight_focal * focal)
        
        return total_loss


# ============================================================================
# VERSION 3: ADVANCED (For experts - Nếu cần squeeze thêm 2-3%)
# ============================================================================

class TverskyLoss(nn.Module):
    """
    Tversky Loss - Điều chỉnh False Positive vs False Negative
    alpha = 0.7: Ưu tiên recall (không bỏ sót răng)
    alpha = 0.3: Ưu tiên precision (không phát hiện nhầm)
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1]  # Foreground only
        targets_fg = (targets == 1).float()
        
        # True Positives, False Positives, False Negatives
        TP = (probs * targets_fg).sum(dim=(1, 2))
        FP = (probs * (1 - targets_fg)).sum(dim=(1, 2))
        FN = ((1 - probs) * targets_fg).sum(dim=(1, 2))
        
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        
        return 1 - tversky.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - Focus vào edge của răng
    Quan trọng cho medical segmentation
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1]
        targets_fg = (targets == 1).float()
        
        # Compute gradients (edge detection)
        dy = targets_fg[:, 1:, :] - targets_fg[:, :-1, :]
        dx = targets_fg[:, :, 1:] - targets_fg[:, :, :-1]
        
        # Boundary map
        boundary = torch.zeros_like(targets_fg)
        boundary[:, 1:, :] += dy.abs()
        boundary[:, :, 1:] += dx.abs()
        boundary = (boundary > 0).float()
        
        # Focus loss on boundaries
        boundary_loss = F.binary_cross_entropy(
            probs, 
            targets_fg,
            weight=boundary * 5 + 1,  # 5x weight on boundaries
            reduction='mean'
        )
        
        return boundary_loss


class CombinedLoss_Advanced(nn.Module):
    """
    ADVANCED: Tversky + Focal + Boundary
    Cho experts muốn push tới giới hạn
    """
    def __init__(self, 
                 weight_tversky=0.4,
                 weight_focal=0.3, 
                 weight_boundary=0.3,
                 alpha=0.5):  # 0.5=balanced, 0.7=recall, 0.3=precision
        super().__init__()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=1-alpha)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.boundary_loss = BoundaryLoss()
        
        self.weight_tversky = weight_tversky
        self.weight_focal = weight_focal
        self.weight_boundary = weight_boundary
    
    def forward(self, logits, targets):
        tversky = self.tversky_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        boundary = self.boundary_loss(logits, targets)
        
        total_loss = (self.weight_tversky * tversky +
                     self.weight_focal * focal +
                     self.weight_boundary * boundary)
        
        return total_loss


# FACTORY - Chọn version nào dùng

def get_loss(version='improved'):
    """
    version: 'basic' | 'improved' | 'advanced'
    
    RECOMMENDED: 
    - Bắt đầu: 'basic' hoặc 'improved'
    - Nếu cần thêm 2-3%: 'advanced'
    """
    if version == 'basic':
        return CombinedLoss_Basic()
    elif version == 'improved':
        return CombinedLoss_Improved()
    elif version == 'advanced':
        return CombinedLoss_Advanced(alpha=0.5)  
    else:
        raise ValueError(f"Unknown version: {version}")