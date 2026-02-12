"""
Phase 8: PyTorch MLP model wrappers.

SoftmaxMLP   -- Predicts proportions via softmax + KL divergence loss.
                Architecture: GeGLU blocks. Best R2 accuracy.

ILR_MLP      -- Predicts 5 ILR coordinates via MSE loss, then inverts
                to proportions. Respects Aitchison geometry, so Aitchison
                distance and compositional bias should improve.

DirichletMLP -- (stretch goal) Predicts Dirichlet concentration parameters
                for both mean predictions and calibrated uncertainty.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# =====================================================================
# GeGLU block
# =====================================================================

class GeGLUBlock(nn.Module):
    """
    GeGLU activation block: GELU(xW₁ + b₁) ⊙ (xW₂ + b₂)

    Splits a double-width linear layer into gate and value paths.
    Includes LayerNorm and Dropout for regularization.
    """

    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.linear(x)
        gate, value = h.chunk(2, dim=-1)
        h = F.gelu(gate) * value
        h = self.norm(h)
        return self.dropout(h)


# =====================================================================
# SoftmaxMLP
# =====================================================================

class _SoftmaxMLPNet(nn.Module):
    """Raw PyTorch network for softmax MLP."""

    def __init__(self, input_dim, n_classes, hidden_dim=256,
                 n_layers=3, dropout=0.15):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(n_layers):
            layers.append(GeGLUBlock(current_dim, hidden_dim, dropout))
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        """Return log-softmax logits (N, C)."""
        h = self.backbone(x)
        return F.log_softmax(self.head(h), dim=-1)

    def predict_proportions(self, x):
        """Return proportions (N, C) — softmax output."""
        h = self.backbone(x)
        return F.softmax(self.head(h), dim=-1)


class SoftmaxMLP:
    """
    MLP with softmax output head and KL divergence loss.

    Predicts land-cover proportions directly on the simplex. No ILR
    transform needed — softmax enforces valid compositions by construction.

    Architecture: Input → [GeGLU Block]×N → Linear → softmax

    Training uses KL(y_true || softmax(logits)) as loss, which treats
    the true proportions as a target distribution.
    """

    def __init__(self, n_classes=6, hidden_dim=256, n_layers=3,
                 dropout=0.15, lr=1e-3, weight_decay=1e-4,
                 batch_size=512, max_epochs=200, patience=15,
                 device=None, random_state=42):
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = StandardScaler()
        self.net = None
        self.train_losses = []
        self.val_losses = []

    def _build_net(self, input_dim):
        torch.manual_seed(self.random_state)
        self.net = _SoftmaxMLPNet(
            input_dim, self.n_classes, self.hidden_dim,
            self.n_layers, self.dropout,
        ).to(self.device)

    def fit(self, X_train, z_train_or_y, X_val=None, z_val_or_y=None):
        """
        Fit on proportion targets (not ILR — softmax handles the simplex).

        Despite the interface naming (z_train_or_y), this model expects
        raw proportions y (N, 6). The z_ prefix in the interface is for
        compatibility with the pipeline that passes ILR targets to other
        models — here we use y directly.

        Parameters
        ----------
        X_train : ndarray (N, D)
        z_train_or_y : ndarray (N, C) — proportions, NOT ILR
        X_val, z_val_or_y : optional validation data
        """
        y_train = z_train_or_y
        y_val = z_val_or_y

        X_scaled = self.scaler.fit_transform(X_train)
        self._build_net(X_scaled.shape[1])

        # Smooth targets to avoid log(0) in KL
        eps = 1e-7
        y_train_s = np.clip(y_train, eps, 1.0)
        y_train_s = y_train_s / y_train_s.sum(axis=1, keepdims=True)

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_train_s, dtype=torch.float32)

        train_ds = TensorDataset(X_t, y_t)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True, drop_last=False)

        # Validation data
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_s = self.scaler.transform(X_val)
            y_val_s = np.clip(y_val, eps, 1.0)
            y_val_s = y_val_s / y_val_s.sum(axis=1, keepdims=True)
            X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val_s, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.max_epochs):
            # -- Train --
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                log_pred = self.net(xb)
                # KL(target || pred) = sum(target * (log(target) - log(pred)))
                loss = F.kl_div(log_pred, yb, reduction="batchmean",
                                log_target=False)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_train_loss)

            # -- Validate --
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    log_pred_val = self.net(X_val_t)
                    val_loss = F.kl_div(log_pred_val, y_val_t,
                                        reduction="batchmean",
                                        log_target=False).item()
                self.val_losses.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.net.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break
            else:
                scheduler.step(avg_train_loss)

        # Restore best model
        if best_state is not None:
            self.net.load_state_dict(best_state)

        return self

    def predict_proportions(self, X):
        """Predict proportions (N, C) — softmax output."""
        self.net.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = self.net.predict_proportions(X_t).cpu().numpy()
        return y_pred

    def predict_ilr(self, X):
        """
        Not naturally ILR-based, but provided for interface compatibility.
        Returns proportions instead (the pipeline handles this).
        """
        return self.predict_proportions(X)

    def get_params_dict(self):
        return {
            "model": "SoftmaxMLP",
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "epochs_trained": len(self.train_losses),
            "device": str(self.device),
        }


# =====================================================================
# DirichletMLP (stretch goal)
# =====================================================================

class _DirichletMLPNet(nn.Module):
    """
    MLP that predicts Dirichlet concentration parameters.

    Output: α = softplus(logits) + 1 (ensures α > 1 for unimodal Dirichlet).

    Uncertainty from the Dirichlet posterior:
        mean:      μ_i = α_i / α0           where α0 = Σα
        variance:  Var[p_i] = α_i(α0 - α_i) / (α0²(α0 + 1))
        precision: α0 (higher = more confident)
    """

    def __init__(self, input_dim, n_classes, hidden_dim=256,
                 n_layers=3, dropout=0.15):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(n_layers):
            layers.append(GeGLUBlock(current_dim, hidden_dim, dropout))
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        """Return concentration parameters α (N, C), all > 1."""
        h = self.backbone(x)
        return F.softplus(self.head(h)) + 1.0

    def mean_and_uncertainty(self, x):
        """
        Returns
        -------
        mean : (N, C) — expected proportions μ_i = α_i / α0
        precision_inv : (N,) — 1/(α0+1), scalar uncertainty (lower = more certain)
        """
        alpha = self.forward(x)
        alpha_sum = alpha.sum(dim=-1, keepdim=True)
        mean = alpha / alpha_sum
        precision_inv = 1.0 / (alpha_sum.squeeze(-1) + 1.0)
        return mean, precision_inv

    def full_uncertainty(self, x):
        """
        Full Dirichlet uncertainty decomposition.

        Returns
        -------
        mean : (N, C) — μ_i = α_i / α0
        variance : (N, C) — Var[p_i] = α_i(α0 - α_i) / (α0²(α0 + 1))
        alpha0 : (N,) — concentration/precision (higher = more confident)
        """
        alpha = self.forward(x)
        alpha0 = alpha.sum(dim=-1, keepdim=True)  # (N, 1)
        mean = alpha / alpha0
        variance = (alpha * (alpha0 - alpha)) / (alpha0 ** 2 * (alpha0 + 1))
        return mean, variance, alpha0.squeeze(-1)


class DirichletMLP:
    """
    MLP with Dirichlet output head.

    Predicts concentration parameters α of a Dirichlet distribution,
    giving both mean proportions and uncertainty from a single forward pass.

    Loss: negative Dirichlet log-likelihood.
    """

    def __init__(self, n_classes=6, hidden_dim=256, n_layers=3,
                 dropout=0.15, lr=1e-3, weight_decay=1e-4,
                 batch_size=512, max_epochs=200, patience=15,
                 device=None, random_state=42):
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = StandardScaler()
        self.net = None
        self.train_losses = []
        self.val_losses = []

    def _build_net(self, input_dim):
        torch.manual_seed(self.random_state)
        self.net = _DirichletMLPNet(
            input_dim, self.n_classes, self.hidden_dim,
            self.n_layers, self.dropout,
        ).to(self.device)

    @staticmethod
    def _dirichlet_nll(alpha, y):
        """
        Negative log-likelihood of Dirichlet distribution.

        NLL = -log B(α) + Σ (α_i - 1) log(y_i)
            = log Γ(Σα) - Σ log Γ(α_i) + Σ (α_i - 1) log(y_i)

        (negated for minimization)
        """
        alpha_sum = alpha.sum(dim=-1)
        nll = (
            -torch.lgamma(alpha_sum)
            + torch.lgamma(alpha).sum(dim=-1)
            - ((alpha - 1.0) * torch.log(y + 1e-7)).sum(dim=-1)
        )
        return nll.mean()

    def fit(self, X_train, z_train_or_y, X_val=None, z_val_or_y=None):
        """Fit on proportion targets (not ILR)."""
        y_train = z_train_or_y
        y_val = z_val_or_y

        X_scaled = self.scaler.fit_transform(X_train)
        self._build_net(X_scaled.shape[1])

        eps = 1e-6
        y_train_s = np.clip(y_train, eps, 1.0 - eps)
        y_train_s = y_train_s / y_train_s.sum(axis=1, keepdims=True)

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_train_s, dtype=torch.float32)

        train_ds = TensorDataset(X_t, y_t)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True, drop_last=False)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_s = self.scaler.transform(X_val)
            y_val_s = np.clip(y_val, eps, 1.0 - eps)
            y_val_s = y_val_s / y_val_s.sum(axis=1, keepdims=True)
            X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val_s, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.max_epochs):
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                alpha = self.net(xb)
                loss = self._dirichlet_nll(alpha, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_train_loss)

            if has_val:
                self.net.eval()
                with torch.no_grad():
                    alpha_val = self.net(X_val_t)
                    val_loss = self._dirichlet_nll(alpha_val, y_val_t).item()
                self.val_losses.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.net.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break
            else:
                scheduler.step(avg_train_loss)

        if best_state is not None:
            self.net.load_state_dict(best_state)

        return self

    def predict_proportions(self, X):
        """Predicted mean proportions α_i / Σα."""
        self.net.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            mean, _ = self.net.mean_and_uncertainty(X_t)
        return mean.cpu().numpy()

    def predict_uncertainty(self, X):
        """Scalar uncertainty per sample: 1 / (alpha0 + 1)."""
        self.net.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, unc = self.net.mean_and_uncertainty(X_t)
        return unc.cpu().numpy()

    def predict_full_uncertainty(self, X):
        """
        Full Dirichlet uncertainty.

        Returns
        -------
        mean : (N, C) — expected proportions
        variance : (N, C) — per-class Dirichlet variance
        alpha0 : (N,) — precision (concentration sum)
        """
        self.net.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            mean, var, alpha0 = self.net.full_uncertainty(X_t)
        return mean.cpu().numpy(), var.cpu().numpy(), alpha0.cpu().numpy()

    def predict_ilr(self, X):
        """Interface compat — returns proportions."""
        return self.predict_proportions(X)

    def get_params_dict(self):
        return {
            "model": "DirichletMLP",
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "epochs_trained": len(self.train_losses),
            "device": str(self.device),
        }


# =====================================================================
# ILR-MLP (compositional-aware)
# =====================================================================

class _ILR_MLPNet(nn.Module):
    """MLP that predicts ILR coordinates directly with MSE loss."""

    def __init__(self, input_dim, ilr_dim, hidden_dim=256,
                 n_layers=3, dropout=0.15):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(n_layers):
            layers.append(GeGLUBlock(current_dim, hidden_dim, dropout))
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, ilr_dim)

    def forward(self, x):
        """Return ILR coordinates (N, D-1)."""
        return self.head(self.backbone(x))


class ILR_MLP:
    """
    MLP trained in ILR space with MSE loss.

    Unlike SoftmaxMLP which predicts proportions directly, this model
    predicts ILR coordinates and inverts them to proportions. This
    respects Aitchison geometry:
    - MSE in ILR space = squared Aitchison distance
    - Training optimizes compositional accuracy directly
    - Expected to have better Aitchison distance and Moran's I
      at possibly slight cost to R2

    Architecture: same GeGLU backbone as SoftmaxMLP.
    Output: 5 ILR coordinates (for 6 classes)
    Loss: MSE(z_pred, z_true) where z = ILR(y)
    Inference: proportions = ILR_inverse(z_pred)
    """

    def __init__(self, n_classes=6, hidden_dim=256, n_layers=3,
                 dropout=0.15, lr=1e-3, weight_decay=1e-4,
                 batch_size=512, max_epochs=200, patience=15,
                 device=None, random_state=42, basis=None):
        self.n_classes = n_classes
        self.ilr_dim = n_classes - 1
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.basis = basis  # ILR basis matrix

        self.scaler = StandardScaler()
        self.net = None
        self.train_losses = []
        self.val_losses = []

    def _build_net(self, input_dim):
        torch.manual_seed(self.random_state)
        self.net = _ILR_MLPNet(
            input_dim, self.ilr_dim, self.hidden_dim,
            self.n_layers, self.dropout,
        ).to(self.device)

    def fit(self, X_train, z_train, X_val=None, z_val=None):
        """
        Fit on ILR-transformed targets (same interface as tree models).

        Parameters
        ----------
        X_train : ndarray (N, D)
        z_train : ndarray (N, C-1) -- ILR coordinates
        X_val, z_val : optional validation data
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self._build_net(X_scaled.shape[1])

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        z_t = torch.tensor(z_train, dtype=torch.float32)

        train_ds = TensorDataset(X_t, z_t)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True, drop_last=False)

        has_val = X_val is not None and z_val is not None
        if has_val:
            X_val_s = self.scaler.transform(X_val)
            X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(self.device)
            z_val_t = torch.tensor(z_val, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.max_epochs):
            self.net.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, zb in train_dl:
                xb, zb = xb.to(self.device), zb.to(self.device)
                z_pred = self.net(xb)
                loss = F.mse_loss(z_pred, zb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_train_loss)

            if has_val:
                self.net.eval()
                with torch.no_grad():
                    z_val_pred = self.net(X_val_t)
                    val_loss = F.mse_loss(z_val_pred, z_val_t).item()
                self.val_losses.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.net.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break
            else:
                scheduler.step(avg_train_loss)

        if best_state is not None:
            self.net.load_state_dict(best_state)

        return self

    def predict_ilr(self, X):
        """Predict ILR coordinates."""
        self.net.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            z_pred = self.net(X_t).cpu().numpy()
        return z_pred

    def predict_proportions(self, X):
        """Predict ILR then invert to simplex proportions."""
        from src.transforms import ilr_inverse
        z_pred = self.predict_ilr(X)
        return ilr_inverse(z_pred, basis=self.basis)

    def get_params_dict(self):
        return {
            "model": "ILR_MLP",
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "epochs_trained": len(self.train_losses),
            "device": str(self.device),
        }
