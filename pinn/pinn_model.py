import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    Simple fully-connected net:
      input: [t, I, Ta, v]   (time, irradiance, ambient temp, wind)
      output: Tp (panel temp)
    """
    def __init__(self, in_dim=4, hidden=128, depth=6, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim if i == 0 else hidden, hidden), act()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def physics_residual(model, tI_Ta_v, alpha=0.03, beta=0.02, gamma=0.02):
    """
    dTp/dt - (alpha*I - beta*(Tp - Ta) - gamma*v)  -> should be ~ 0
    alpha,beta,gamma can be tuned; start modest to stabilize training.
    """
    tI_Ta_v = tI_Ta_v.clone().requires_grad_(True)
    Tp = model(tI_Ta_v)                                    # [N,1]
    dTp_dt = torch.autograd.grad(
        Tp, tI_Ta_v, grad_outputs=torch.ones_like(Tp),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]  # derivative w.r.t. column 0 = time

    I   = tI_Ta_v[:, 1:2]
    Ta  = tI_Ta_v[:, 2:3]
    v   = tI_Ta_v[:, 3:4]

    eq = dTp_dt - (alpha * I - beta * (Tp - Ta) - gamma * v)
    return eq
