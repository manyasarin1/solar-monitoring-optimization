import os
import numpy as np
import deepxde as dde
from deepxde.backend import tf

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data_ready")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "runs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n🚀 Starting PINN training for all datasets in data_ready...\n")

# ----------------------------------------------------------------------
# Loop through all .npz files
# ----------------------------------------------------------------------
for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".npz"):
        continue

    city, quarter, scenario = fname.replace(".npz", "").split("_")
    label = f"{city.upper()} {quarter.upper()} — {scenario.upper()}"
    print(f"⚡ Training PINN for {label}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    data = np.load(os.path.join(DATA_DIR, fname))
    X = data["X"]
    y = data["y"]
    print(f"📦 {fname}: X={X.shape}, y={y.shape}")

    # ------------------------------------------------------------------
    # Define geometry (safe for any dimension)
    # ------------------------------------------------------------------
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)

    print("Feature ranges:")
    for i in range(len(xmin)):
        delta = xmax[i] - xmin[i]
        if delta <= 0:
            print(f"⚠️  Column {i} constant; expanding artificially.")
            xmin[i] -= 1e-3
            xmax[i] += 1e-3
        else:
            print(f"  dim {i}: Δ={delta}")

    geom = dde.geometry.Hypercube(xmin, xmax)
    print(f"   Geometry OK: xmin={xmin}, xmax={xmax}")

    # ------------------------------------------------------------------
    # 🧠 DEFINE THE SOLAR ENERGY-BALANCE PDE  ← main change starts here
    # ------------------------------------------------------------------
    def pde(x, y):
        """
        Physics: Energy balance on a solar panel surface.
        Cp * dTp/dt = G*(1 - ηopt) - hc*(Tp - Ta) - σ*ε*(Tp^4 - Ta^4)
        """
        # Predicted outputs
        Tp_pred = y[:, 0:1]    # panel temperature
        eta_pred = y[:, 1:2]   # panel efficiency

        # Extract relevant inputs (columns from X)
        # [solar_radiation, air_temp, wind_speed, pressure]
        G = x[:, 0:1]
        Ta = x[:, 1:2]

        # Physical constants (approximate)
        Cp = 900.0        # J/kg·K - effective heat capacity
        hc = 0.02         # W/m²·K - convective heat coefficient
        sigma = 5.67e-8   # W/m²·K⁴ - Stefan–Boltzmann constant
        eps = 0.9         # emissivity
        eta_opt = 0.15    # optical efficiency

        # Compute dTp/dt via automatic differentiation
        dTp_dt = dde.grad.jacobian(y, x, i=0, j=0)  # derivative wrt first input (G)
        # You can later adjust j depending on which input represents time if added

        # Energy balance residual
        residual = Cp * dTp_dt - (
            G * (1 - eta_opt)
            - hc * (Tp_pred - Ta)
            - sigma * eps * (Tp_pred**4 - Ta**4)
        )
        return [residual]
    # ------------------------------------------------------------------
    # 🧠 END OF PDE DEFINITION
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Define and train model
    # ------------------------------------------------------------------
    net = dde.nn.FNN([4, 64, 64, 64, 2], "tanh", "Glorot normal")
    data_obj = dde.data.PDE(geom, pde, [], num_domain=1000)
    model = dde.Model(data_obj, net)
    model.compile("adam", lr=1e-3)
    losshistory, train_state = model.train(epochs=1000, display_every=200)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    run_dir = os.path.join(OUTPUT_DIR, fname.replace(".npz", ""))
    os.makedirs(run_dir, exist_ok=True)
    dde.saveplot(losshistory, train_state, issave=False, isplot=False)
    np.savez(
        os.path.join(run_dir, "results.npz"),
        X=X,
        y_true=y,
        y_pred=model.predict(X),
    )

    print(f"✅ Finished training {label}")
    print("-" * 60)

print("\n🎉 All PINN trainings complete! Check results in 'runs/' directory.\n")
