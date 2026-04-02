import os
import sys
import cv2
import torch
import numpy as np


# MOCK MAMBA 
class MockMamba:
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        self.d_model = d_model

    def __call__(self, x):
        return x

try:
    from mamba_ssm import Mamba
except ImportError:
    print("⚠️  mamba_ssm not installed, using mock")
    sys.modules['mamba_ssm'] = type(
        'MockModule', (), {'Mamba': MockMamba}
    )()

# IMPORT MODEL
from models.mamba_unet import create_mamba_unet

# CONFIG
IMG_SIZE = 512
NUM_CLASSES = 2
IN_CHANS = 1
DEPTHS= [2, 2, 2, 1]
EMBED_DIM= 32

CHECKPOINT_PATH = "checkpoints/20260305_205950/best.pth"
TEST_IMAGE_PATH = "./21.JPG"

SAVE_MASK_PATH    = "prediction_mask10.png"
SAVE_OVERLAY_PATH = "prediction_overlay10.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# UTILS

def print_shape(name, x):
    if isinstance(x, torch.Tensor):
        print(f"  {name:30s}: {tuple(x.shape)}")
    else:
        print(f"  {name:30s}: {x}")


def load_image_gray(path, size=512):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f" Cannot read image: {path}"
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    return img


def overlay_mask(image_gray, mask):
    image_gray = (image_gray * 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    overlay = image_rgb.copy()
    overlay[mask > 0] = (0, 0, 255)

    blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
    return blended


# 1. ARCHITECTURE TEST
def test_architecture():
    print("\n" + "="*80)
    print(" TEST ARCHITECTURE")
    print("="*80)

    model = create_mamba_unet(
        in_chans=IN_CHANS,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        depths=DEPTHS,           
        embed_dim=EMBED_DIM,
        
    ).to(DEVICE)

    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, IN_CHANS, IMG_SIZE, IMG_SIZE).to(DEVICE)

    print("\n INPUT")
    print_shape("Input", x)

    # Patch partition
    x_pp = model.patch_partition(x)
    print("\n Patch Partition")
    print_shape("After patch partition", x_pp)

    expected = (batch_size, 32, 128, 128)
    assert x_pp.shape == expected, f" PatchPartition expected {expected}, got {x_pp.shape}"

    # Encoder
    print("\n ENCODER")
    skip_connections = []
    x_enc = x_pp

    for i, stage in enumerate(model.encoder_stages):
        print(f"\nStage {i+1}")
        x_skip, x_enc = stage(x_enc)
        skip_connections.append(x_skip)

        print_shape("Skip", x_skip)
        print_shape("Down", x_enc if i < 3 else x_skip)

    # Bottleneck
    print("\n BOTTLENECK")
    x_bn = model.bottleneck(x_enc)
    print_shape("Bottleneck", x_bn)

    expected_bn = (batch_size, EMBED_DIM * 8, 16, 16)
    assert x_bn.shape == expected_bn, f" Bottleneck expected {expected_bn}, got {x_bn.shape}"

    # Decoder
    print("\n DECODER")
    x_dec = x_bn
    skip_connections = skip_connections[:-1]

    for i, stage in enumerate(model.decoder_stages):
        skip = skip_connections[-(i + 1)]
        print(f"\nDecoder {i+1}")
        print_shape("Input", x_dec)
        print_shape("Skip", skip)

        x_dec = stage(x_dec, skip)
        print_shape("Output", x_dec)

    # Final
    print("\n OUTPUT")
    x_final = model.final_expand(x_dec)
    x_out = model.seg_head(x_final)

    print_shape("Final expand", x_final)
    print_shape("Seg output", x_out)

    expected_out = (batch_size, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
    assert x_out.shape == expected_out, f" Output expected {expected_out}, got {x_out.shape}"

    print("\n Architecture test PASSED!")


# 2. INFERENCE TEST
def test_inference():
    print("\n" + "="*80)
    print(" TEST INFERENCE")
    print("="*80)

    assert os.path.exists(CHECKPOINT_PATH), " Checkpoint not found!"
    assert os.path.exists(TEST_IMAGE_PATH), " Test image not found!"

    print(" Loading model...")
    model = create_mamba_unet(
        in_chans=IN_CHANS,
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        depths=DEPTHS,
        embed_dim=EMBED_DIM,
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    print("Checkpoint type:", type(checkpoint))

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(" Model loaded!")

    # Load image
    img = load_image_gray(TEST_IMAGE_PATH, IMG_SIZE)
    input_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

    print_shape("Input tensor", input_tensor)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1)
        target_prob = probs[:, 1]
        pred_mask = (target_prob > 0.5).float()

    pred_mask_np = pred_mask.squeeze().cpu().numpy()

    # Save mask
    cv2.imwrite(SAVE_MASK_PATH, (pred_mask_np * 255).astype(np.uint8))

    # Overlay
    overlay = overlay_mask(img, pred_mask_np)
    cv2.imwrite(SAVE_OVERLAY_PATH, overlay)

    print("\n Saved:")
    print(f"  - Mask    : {SAVE_MASK_PATH}")
    print(f"  - Overlay : {SAVE_OVERLAY_PATH}")

    # Sanity check
    unique_values = np.unique(pred_mask_np)
    print("\n Mask values:", unique_values)

    assert pred_mask_np.shape == (IMG_SIZE, IMG_SIZE), " Mask shape incorrect"
    assert len(unique_values) <= 2, " Mask is not binary"

    print("\n Inference test PASSED!")


# MAIN
if __name__ == "__main__":
    try:
        test_architecture()
        test_inference()
        print("\n ALL TESTS PASSED SUCCESSFULLY!")

    except Exception as e:
        print("\n TEST FAILED:", e)
        import traceback
        traceback.print_exc()