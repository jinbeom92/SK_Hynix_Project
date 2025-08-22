import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.yaml_config import load_config
from models.hdn import HDNSystem
from data.dataset import _load_sino_uaD

cfg = load_config("config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HDNSystem(cfg).to(device)
ckpt = torch.load("results/best_shared.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

sino = _load_sino_uaD("1.npy")
U, A, D = sino.shape
X = Y = U

pred_vol = np.zeros((X, Y, D), dtype=np.float32)
with torch.no_grad():
    for d in range(D):
        s_slice = torch.from_numpy(sino[:, :, d]).unsqueeze(0).to(device)
        r_hat = model(s_slice, v_slice=None, train_mode=False)
        pred_vol[:, :, d] = r_hat[0, 0].cpu().numpy()
        
arr = pred_vol

print("Shape:", arr.shape)
print("Dtype:", arr.dtype)
print("Min:", arr.min(), "Max:", arr.max())
print("Sample slice [0]:\n", arr)
np.save("predicted_volume", pred_vol)


mid = D // 2
plt.figure(figsize=(5,5))
plt.title(f"Reconstruction slice {mid}")
plt.imshow(pred_vol[:, :, mid], cmap="gray", vmin=0.0, vmax=1.0)
plt.axis("off")
plt.tight_layout()
plt.show()
