import torch
from mamba_ssm import Mamba

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = Mamba(
        d_model=256,
        d_state=16,
        d_conv=4,
        expand=2
    ).to(device)

    x = torch.randn(2, 128, 256, device=device)

    with torch.no_grad():
        y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("✅ Mamba chạy OK")

if __name__ == "__main__":
    main()