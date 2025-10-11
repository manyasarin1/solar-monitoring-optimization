import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pinn_model import PINN, physics_residual
from utils_pinn import load_npz, make_time, normalize_train, apply_norm, weak_label_tp, to_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data_ready"
OUT_DIR = "runs"
os.makedirs(OUT_DIR, exist_ok=True)

# ----- training hyperparams -----
LR = 1e-3
EPOCHS = 3000
PHYS_W = 1.0      # weight for physics loss
WEAK_W = 0.2      # weight for weak-label loss (can set 0 if you want pure physics)
PRINT_EVERY = 200

def train_one_file(npz_path):
    name = os.path.basename(npz_path).replace(".npz", "")
    print(f"\n==== Training on {name} ====")

    # Load data: columns assumed order [I, T2M, WS10M, PS, WSC]
    arr = load_npz(npz_path)
    I   = arr[:, 0:1]
    Ta  = arr[:, 1:2]
    v   = arr[:, 2:3]
    # PS, WSC are available if you want to extend the model inputs later

    n = len(arr)
    t  = make_time(n, dt_hours=1.0)  # naive time index
    X  = np.hstack([t, I, Ta, v])    # [t, I, Ta, v]

    # Normalize inputs (time & features)
    Xn, xstats = normalize_train(X)

    # Weak label for Tp to guide training (can be set WEAK_W=0 to turn off)
    Tp_weak = weak_label_tp(I.squeeze(1), Ta.squeeze(1), v.squeeze(1))  # [N,1]

    # Tensors
    Xn_t, Tp_weak_t = to_tensor(Xn, Tp_weak, device=DEVICE)

    # Model
    model = PINN(in_dim=4, hidden=128, depth=6).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    # Training
    for ep in range(1, EPOCHS + 1):
        opt.zero_grad()

        Tp_pred = model(Xn_t)                               # data-driven head
        # physics
        phys_eq = physics_residual(model, Xn_t, alpha=0.03, beta=0.02, gamma=0.02)
        loss_phys = torch.mean(phys_eq**2)

        # weak label supervision
        loss_weak = mse(Tp_pred, Tp_weak_t)

        loss = PHYS_W * loss_phys + WEAK_W * loss_weak
        loss.backward()
        opt.step()

        if ep % PRINT_EVERY == 0 or ep == 1:
            print(f"[{ep:4d}] total={loss.item():.5f}  phys={loss_phys.item():.5f}  weak={loss_weak.item():.5f}")

    # Save model + quick plot
    mdl_dir = os.path.join(OUT_DIR, name)
    os.makedirs(mdl_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(mdl_dir, "pinn.pt"))
    np.save(os.path.join(mdl_dir, "xstats.npy"), xstats, allow_pickle=True)

    # Plot: Tp_pred vs weak label (as a sanity curve)
    with torch.no_grad():
        Tp_pred_np = Tp_pred.detach().cpu().numpy()
    plt.figure()
    plt.plot(Tp_weak.squeeze(), label="weak Tp", linewidth=1)
    plt.plot(Tp_pred_np.squeeze(), label="PINN Tp", linewidth=1)
    plt.title(f"{name}: Tp (weak vs PINN)")
    plt.xlabel("t (samples)")
    plt.ylabel("Temperature (°C, relative scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(mdl_dir, "tp_curve.png"))
    plt.close()

    # Save a CSV for judges
    out_csv = os.path.join(mdl_dir, "tp_preds.csv")
    np.savetxt(out_csv, np.hstack([Tp_weak, Tp_pred_np]), delimiter=",", header="tp_weak,tp_pinn", comments="")
    print(f"✅ Saved: {mdl_dir}")

def main():
    # choose a small demo subset first (you can broaden later)
    # e.g., Chennai Q1 across 4 scenarios
    subset = [
        "chennai_q1_clean.npz",
        "chennai_q1_sparse.npz",
        "chennai_q1_noisy.npz",
        "chennai_q1_rural.npz",
    ]
    # If you want to train all files, uncomment next line:
    # subset = [os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.npz"))]

    for base in subset:
        npz_path = os.path.join(DATA_DIR, base)
        if os.path.exists(npz_path):
            train_one_file(npz_path)
        else:
            print(f"⚠️ Skipping missing file: {npz_path}")

if __name__ == "__main__":
    main()

