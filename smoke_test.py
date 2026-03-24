
import torch

print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

import pytorch3d
print("pytorch3d import: OK")

from pytorch3d.ops import knn_points
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(1, 64, 3, device=device)
y = torch.randn(1, 128, 3, device=device)
dists, idx, _ = knn_points(x, y, K=3)
print("pytorch3d knn_points: OK", dists.shape, idx.shape, "device:", dists.device)

print("All smoke tests passed.")

