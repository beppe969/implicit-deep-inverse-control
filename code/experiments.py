"""Reproducibility script for

    Implicit Input--Output Model Inversion for Robust Nonlinear Control
    (revised version)

Usage:  python experiments.py [static|narx|servo|all]

The script regenerates every table entry, every figure, and the JSON data
used in the paper.  Compared with the first version, it additionally runs

  * the control-affine ablation (B_u = 0),
  * a cyclic implicit forward model (A != 0) trained with unrolled
    fixed-point iterations,
  * the inverse-sensitivity penalty ablation (lambda_s in {0, 1e-4, 1e-3}),
  * an online Newton root-finding controller on the forward model,
  * the slope-refined certificate (diagonal slope matrix Lambda),
  * the trajectory-wise conformal model-error envelope,
  * a batch of plants that violate the assumed uncertainty envelope,
  * measurement noise in closed loop, and paired bootstrap confidence
    intervals for all closed-loop comparisons.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

# Keep text in vector figures as embedded TrueType outlines rather than Type 3
# bitmap fonts, which improves archival and publisher compatibility.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

SEED = 20260831
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# One thread is markedly faster for the small dense matrices used here.
torch.set_num_threads(1)
DTYPE = torch.float32


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def max_row_sum(M: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.sum(torch.abs(M), dim=1))


def pf_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(M, dtype=float)))))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, nboot: int = 20000, seed: int = 0) -> dict:
    """95% bootstrap CI for mean(a-b) over paired samples (common random numbers)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    idx = rng.integers(0, n, (nboot, n))
    means = d[idx].mean(axis=1)
    return {
        "mean_difference": float(d.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "fraction_negative": float(np.mean(d < 0)),
    }


# -----------------------------------------------------------------------------
# Implicit forward model with exact algebraic inverse
# -----------------------------------------------------------------------------
class AlgebraicImplicitMap(nn.Module):
    """Implicit residual predictor with an exact algebraic inverse.

        h = tanh(A h + B_chi chi + B_u u + b),
        y = base(chi) + D_chi chi + C h + D_u u + c.

    * ``kappa_f = 0`` gives a feedforward hidden block (A = 0).  For
      ``kappa_f > 0`` the forward hidden matrix A is trainable and smoothly
      scaled so that ||A||_inf < kappa_f; the forward equilibrium is then
      computed by fixed-point iteration (unrolled during training).
    * C is smoothly normalized so that ||A - B_u D_u^{-1} C||_inf < kappa_i,
      which enforces the inverse contraction budget.
    * ``input_features=False`` removes B_u (control-affine ablation).
    """

    def __init__(
        self,
        qdim: int,
        udim: int,
        hdim: int,
        D_u: np.ndarray | torch.Tensor,
        D_chi: Optional[np.ndarray | torch.Tensor] = None,
        kappa_i: float = 0.8,
        base_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_features: bool = True,
        kappa_f: float = 0.0,
        train_unroll: int = 20,
    ) -> None:
        super().__init__()
        self.qdim = int(qdim)
        self.udim = int(udim)
        self.hdim = int(hdim)
        self.kappa_i = float(kappa_i)
        self.kappa_f = float(kappa_f)
        self.train_unroll = int(train_unroll)
        self.input_features = bool(input_features)
        D_u_t = torch.as_tensor(D_u, dtype=DTYPE)
        if D_u_t.shape != (udim, udim):
            raise ValueError(f"D_u must have shape {(udim, udim)}, got {tuple(D_u_t.shape)}")
        if D_chi is None:
            D_chi_t = torch.zeros((udim, qdim), dtype=DTYPE)
        else:
            D_chi_t = torch.as_tensor(D_chi, dtype=DTYPE)
            if D_chi_t.shape != (udim, qdim):
                raise ValueError(f"D_chi must have shape {(udim, qdim)}, got {tuple(D_chi_t.shape)}")
        self.register_buffer("D_u", D_u_t)
        self.register_buffer("D_u_inv", torch.linalg.inv(D_u_t))
        self.register_buffer("D_chi", D_chi_t)
        self.base_fn = base_fn
        self.B_chi = nn.Parameter(0.24 * torch.randn(hdim, qdim)) if qdim else None
        if self.input_features:
            self.B_u = nn.Parameter(0.28 * torch.randn(hdim, udim))
        else:
            self.register_buffer("B_u", torch.zeros(hdim, udim))
        self.C_raw = nn.Parameter(0.16 * torch.randn(udim, hdim))
        self.b = nn.Parameter(torch.zeros(hdim))
        self.c = nn.Parameter(torch.zeros(udim))
        if self.kappa_f > 0:
            self.A_raw = nn.Parameter(0.12 * torch.randn(hdim, hdim))
        else:
            self.A_raw = None

    # -- realization matrices -------------------------------------------------
    def A_matrix(self) -> Optional[torch.Tensor]:
        if self.A_raw is None:
            return None
        nr = max_row_sum(self.A_raw)
        scale = self.kappa_f / (self.kappa_f + nr + 1e-8)
        return self.A_raw * scale

    def effective_C(self) -> torch.Tensor:
        A = self.A_matrix()
        budget = self.kappa_i - (max_row_sum(A) if A is not None else 0.0)
        if not self.input_features:
            return self.C_raw
        raw_loop = -self.B_u @ self.D_u_inv @ self.C_raw
        nr = max_row_sum(raw_loop)
        # smooth scaling: ||A_i||_inf <= ||A||_inf + budget*nr/(budget+nr) < kappa_i
        scale = budget / (budget + nr + 1e-8)
        return self.C_raw * scale

    def inverse_A(self) -> torch.Tensor:
        A_i = -self.B_u @ self.D_u_inv @ self.effective_C()
        A = self.A_matrix()
        return A_i if A is None else A_i + A

    def base(self, chi: torch.Tensor) -> torch.Tensor:
        if self.base_fn is None:
            return torch.zeros((chi.shape[0], self.udim), dtype=chi.dtype, device=chi.device)
        return self.base_fn(chi)

    # -- forward evaluation ---------------------------------------------------
    def _forward_affine(self, chi: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        a = u @ self.B_u.T + self.b
        if self.qdim:
            a = a + chi @ self.B_chi.T
        return a

    def hidden_forward(self, chi: torch.Tensor, u: torch.Tensor, tol: float = 1e-8, max_iter: int = 200) -> torch.Tensor:
        a = self._forward_affine(chi, u)
        A = self.A_matrix()
        if A is None:
            self.last_forward_iters = 1
            return torch.tanh(a)
        h = torch.zeros_like(a)
        if self.training:
            for _ in range(self.train_unroll):
                h = torch.tanh(h @ A.T + a)
            return h
        with torch.no_grad():
            for k in range(max_iter):
                h_new = torch.tanh(h @ A.T + a)
                if torch.max(torch.abs(h_new - h)) <= tol:
                    h = h_new
                    break
                h = h_new
        self.last_forward_iters = k + 1
        return h

    def forward(self, chi: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        h = self.hidden_forward(chi, u)
        y = h @ self.effective_C().T + u @ self.D_u.T + self.c
        if self.qdim:
            y = y + chi @ self.D_chi.T
        if self.base_fn is not None:
            y = y + self.base(chi)
        return y

    # -- inverse evaluation ---------------------------------------------------
    @torch.no_grad()
    def _inverse_affine(self, chi: torch.Tensor, y: torch.Tensor):
        y_bar = y - self.c
        if self.qdim:
            y_bar = y_bar - chi @ self.D_chi.T
        if self.base_fn is not None:
            y_bar = y_bar - self.base(chi)
        rhs = y_bar @ (self.B_u @ self.D_u_inv).T + self.b
        if self.qdim:
            rhs = rhs + chi @ self.B_chi.T
        return y_bar, rhs

    @torch.no_grad()
    def inverse(
        self,
        chi: torch.Tensor,
        y: torch.Tensor,
        tol: float = 1e-7,
        max_iter: int = 100,
        warm: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
        C = self.effective_C()
        A_i = self.inverse_A()
        y_bar, rhs = self._inverse_affine(chi, y)
        h = torch.zeros((y.shape[0], self.hdim), dtype=y.dtype, device=y.device) if warm is None else warm.clone()
        its = torch.zeros(y.shape[0], dtype=torch.int64, device=y.device)
        active = torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
        for k in range(max_iter):
            h_new = torch.tanh(h @ A_i.T + rhs)
            r = torch.max(torch.abs(h_new - h), dim=1).values
            done = active & (r <= tol)
            its[done] = k + 1
            active &= ~done
            h = h_new
            if not torch.any(active):
                break
        its[active] = max_iter
        u = (y_bar - h @ C.T) @ self.D_u_inv.T
        if return_info:
            z = h @ A_i.T + rhs
            fp = torch.max(torch.abs(h - torch.tanh(z)), dim=1).values
            return u, {
                "h": h,
                "z": z,  # inverse pre-activation, used for slope bounds
                "iterations": its,
                "residual": fp,
                "kappa_inf": float(max_row_sum(A_i).item()),
            }
        return u

    def sensitivity_penalty(self) -> torch.Tensor:
        """Differentiable ||S_chi||_inf with Lambda = I (used as Psi_s in training)."""
        C = self.effective_C()
        A_i = self.inverse_A()
        I = torch.eye(self.hdim, dtype=C.dtype)
        R_i = torch.linalg.inv(I - torch.abs(A_i))
        B_i_chi = self.B_chi - self.B_u @ self.D_u_inv @ self.D_chi
        S_chi = torch.abs(self.D_u_inv @ self.D_chi) + torch.abs(self.D_u_inv @ C) @ R_i @ torch.abs(B_i_chi)
        return max_row_sum(S_chi)

    # -- componentwise certificate matrices -----------------------------------
    @torch.no_grad()
    def sensitivity_matrices(self, lam: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
        """Componentwise sensitivity/certificate matrices.

        ``lam`` is the diagonal of the slope matrix Lambda (defaults to ones,
        i.e. the plain nonexpansiveness bound).  With R_i = (I - Lambda|A_i|)^{-1}
        and R_f = (I - Lambda|A|)^{-1}:
            S_chi = |Du^-1 Dchi| + |Du^-1 C| R_i Lambda |B_ichi|
            S_y   = |Du^-1|      + |Du^-1 C| R_i Lambda |B_iy|
            K_u   = |Du^-1 C| R_i,     K_y = |C| R_f.
        """
        C = self.effective_C().cpu().numpy().astype(float)
        A_i = self.inverse_A().cpu().numpy().astype(float)
        A = self.A_matrix()
        A = np.zeros((self.hdim, self.hdim)) if A is None else A.cpu().numpy().astype(float)
        Du_inv = self.D_u_inv.cpu().numpy().astype(float)
        Dchi = self.D_chi.cpu().numpy().astype(float)
        Bu = self.B_u.cpu().numpy().astype(float)
        Bchi = self.B_chi.cpu().numpy().astype(float) if self.qdim else np.zeros((self.hdim, 0))
        lam = np.ones(self.hdim) if lam is None else np.asarray(lam, dtype=float)
        Lam = np.diag(lam)
        I = np.eye(self.hdim)
        R_i = np.linalg.inv(I - Lam @ np.abs(A_i))
        R_f = np.linalg.inv(I - Lam @ np.abs(A))
        B_i_chi = Bchi - Bu @ Du_inv @ Dchi
        B_i_y = Bu @ Du_inv
        DC = np.abs(Du_inv @ C)
        S_chi = np.abs(Du_inv @ Dchi) + DC @ R_i @ Lam @ np.abs(B_i_chi)
        S_y = np.abs(Du_inv) + DC @ R_i @ Lam @ np.abs(B_i_y)
        K_u = DC @ R_i
        K_y = np.abs(C) @ R_f
        return {
            "A": A,
            "A_i": A_i,
            "B_i_chi": B_i_chi,
            "B_i_y": B_i_y,
            "R_i": R_i,
            "R_f": R_f,
            "S_chi": S_chi,
            "S_y": S_y,
            "K_u": K_u,
            "K_y": K_y,
            "rho_absA": pf_radius(np.abs(A)),
            "rho_absAi": pf_radius(np.abs(A_i)),
            "lam": lam,
        }


class StandardizedMLP(nn.Module):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, width: int = 64, depth: int = 3):
        super().__init__()
        self.register_buffer("xm", X.mean(0))
        self.register_buffer("xs", X.std(0).clamp_min(1e-4))
        self.register_buffer("ym", Y.mean(0))
        self.register_buffer("ys", Y.std(0).clamp_min(1e-4))
        layers = []
        d = X.shape[1]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(d, width), nn.Tanh()])
            d = width
        layers.append(nn.Linear(d, Y.shape[1]))
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net((X - self.xm) / self.xs) * self.ys + self.ym


def fit_forward(
    model: AlgebraicImplicitMap,
    chi: torch.Tensor,
    u: torch.Tensor,
    y: torch.Tensor,
    steps: int = 1800,
    batch: int = 1024,
    lr: float = 2e-3,
    wd: float = 2e-6,
    sensitivity_weight: float = 0.0,
) -> None:
    """Adam fit of the forward predictor.  Checkpoint selection uses the
    training MSE on a fixed subset of at most 3000 training samples (the same
    rule is used for the direct-inverse baseline)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = float("inf")
    best_state = None
    n = len(chi)
    model.train()
    for k in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        pred = model(chi[idx], u[idx])
        loss = torch.mean((pred - y[idx]) ** 2)
        if sensitivity_weight > 0 and model.qdim:
            # Psi_s = ||S_chi||_inf, the exact componentwise inverse-sensitivity
            # bound of Proposition 7, differentiated through the resolvent.
            loss = loss + sensitivity_weight * model.sensitivity_penalty()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (k + 1) % 40 == 0:
            model.eval()
            with torch.no_grad():
                sel = torch.arange(min(n, 3000))
                val = float(torch.mean((model(chi[sel], u[sel]) - y[sel]) ** 2).item())
            model.train()
            if val < best:
                best = val
                best_state = {a: b.detach().clone() for a, b in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()


def fit_mlp(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    steps: int = 1800,
    batch: int = 1024,
    lr: float = 2e-3,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-6)
    best = float("inf")
    best_state = None
    n = len(X)
    for k in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        pred = model(X[idx])
        loss = torch.mean((pred - Y[idx]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (k + 1) % 40 == 0:
            with torch.no_grad():
                sel = torch.arange(min(n, 3000))
                val = float(torch.mean((model(X[sel]) - Y[sel]) ** 2).item())
            if val < best:
                best = val
                best_state = {a: b.detach().clone() for a, b in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)


# -----------------------------------------------------------------------------
# Online Newton root finding on the forward model (baseline controller)
# -----------------------------------------------------------------------------
@torch.no_grad()
def newton_inverse(model: AlgebraicImplicitMap, chi: torch.Tensor, v: torch.Tensor, u0: torch.Tensor, tol: float = 2e-7, max_iter: int = 25, stall: float = 1e-7):
    """Batched Newton iteration for M_chi(u) = v on the forward model.

    Returns the input, the Newton iteration count per sample and the number
    of hidden-layer (tanh) evaluations per sample: each Newton step costs one
    forward evaluation (one hidden-layer pass for A = 0, a fixed-point solve
    for A != 0) plus an m x m linear solve (and an n_xi x n_xi solve if A != 0).
    The Jacobian is exact: dM/du = D_u + C (I - diag(s) A)^{-1} diag(s) B_u,
    s = 1 - h^2."""
    u = u0.clone()
    m = u.shape[1]
    C = model.effective_C()
    A = model.A_matrix()
    its = torch.zeros(u.shape[0], dtype=torch.int64)
    hidden_evals = torch.zeros(u.shape[0], dtype=torch.int64)
    active = torch.ones(u.shape[0], dtype=torch.bool)
    for k in range(max_iter):
        h = model.hidden_forward(chi, u)
        hidden_evals[active] += model.last_forward_iters
        y = h @ C.T + u @ model.D_u.T + model.c
        if model.qdim:
            y = y + chi @ model.D_chi.T
        if model.base_fn is not None:
            y = y + model.base(chi)
        res = y - v
        rn = torch.max(torch.abs(res), dim=1).values
        done = active & (rn <= tol)
        its[done] = k
        active &= ~done
        if not torch.any(active):
            break
        s_ = 1.0 - h**2
        Bu = model.B_u
        if A is None:
            Jh = s_.unsqueeze(2) * Bu.unsqueeze(0)
        else:
            N = u.shape[0]
            I = torch.eye(model.hdim, dtype=u.dtype).expand(N, -1, -1)
            M = I - s_.unsqueeze(2) * A.unsqueeze(0)
            Jh = torch.linalg.solve(M, s_.unsqueeze(2) * Bu.unsqueeze(0))
        Jac = model.D_u.unsqueeze(0) + torch.einsum("ij,njk->nik", C, Jh)
        step = torch.linalg.solve(Jac, res.unsqueeze(2)).squeeze(2)
        stalled = active & (torch.max(torch.abs(step), dim=1).values <= stall)
        u = torch.where(active.unsqueeze(1), u - step, u)
        its[stalled] = k + 1
        active &= ~stalled
        if not torch.any(active):
            break
    its[active] = max_iter
    return u, its, hidden_evals


# -----------------------------------------------------------------------------
# Experiment 1: static nonlinear MIMO map
# -----------------------------------------------------------------------------
def true_static(u: torch.Tensor) -> torch.Tensor:
    D = torch.tensor([[1.0, 0.18], [-0.12, 0.92]], dtype=u.dtype)
    y = u @ D.T
    y1 = 0.34 * torch.sin(0.85 * u[:, 0] + 0.25 * u[:, 1]) + 0.11 * torch.tanh(u[:, 0] - 0.7 * u[:, 1])
    y2 = 0.27 * torch.tanh(0.5 * u[:, 0] + 0.9 * u[:, 1]) + 0.09 * torch.sin(u[:, 0] * u[:, 1])
    return y + torch.stack([y1, y2], 1)


def run_static() -> dict:
    torch.manual_seed(SEED + 1)
    n = 7000
    u = 4 * torch.rand(n, 2) - 2
    y0 = true_static(u)
    y = y0 + 0.09 * torch.randn_like(y0)
    perm = torch.randperm(n)
    tr, te = perm[:5500], perm[5500:]
    Xaug = torch.cat([u[tr], torch.ones(len(tr), 1)], 1)
    coef = torch.linalg.lstsq(Xaug, y[tr]).solution
    Dhat = coef[:2].T.detach()
    chi = torch.zeros(n, 0)
    model = AlgebraicImplicitMap(0, 2, 40, Dhat, kappa_i=0.78)
    fit_forward(model, chi[tr], u[tr], y[tr], steps=1500, batch=1024, lr=2.7e-3)
    inv = StandardizedMLP(y[tr], u[tr], width=64, depth=3)
    fit_mlp(inv, y[tr], u[tr], steps=1600, batch=1024, lr=2.5e-3)

    with torch.no_grad():
        fwd = model(chi[te], u[te])
        ua, info = model.inverse(chi[te], y0[te], return_info=True)
        ud = inv(y0[te])
        model_cycle = model(chi[te], ua)
        true_cycle_a = true_static(ua)
        true_cycle_d = true_static(ud)
        yn = y0[te] + 0.05 * torch.randn_like(y0[te])
        una = model.inverse(chi[te], yn)
        und = inv(yn)
        # Newton root finding on the forward model for the same targets
        t0 = time.perf_counter()
        un, nits, nevals = newton_inverse(model, chi[te], y0[te], torch.zeros_like(u[te]), tol=5e-7, stall=5e-7)
        t_newton = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = model.inverse(chi[te], y0[te])
        t_alg = time.perf_counter() - t0

    out = {
        "forward_rmse": rms((fwd - y0[te]).numpy()),
        "inverse_rmse_algebraic": rms((ua - u[te]).numpy()),
        "inverse_rmse_direct": rms((ud - u[te]).numpy()),
        "inverse_rmse_newton": rms((un - u[te]).numpy()),
        "learned_model_cycle_rmse": rms((model_cycle - y0[te]).numpy()),
        "true_cycle_rmse_algebraic": rms((true_cycle_a - y0[te]).numpy()),
        "true_cycle_rmse_direct": rms((true_cycle_d - y0[te]).numpy()),
        "noisy_inverse_rmse_algebraic": rms((una - u[te]).numpy()),
        "noisy_inverse_rmse_direct": rms((und - u[te]).numpy()),
        "mean_iterations": float(info["iterations"].float().mean().item()),
        "p95_iterations": float(torch.quantile(info["iterations"].float(), 0.95).item()),
        "max_iterations": float(info["iterations"].max().item()),
        "newton_mean_iterations": float(nits.float().mean().item()),
        "newton_mean_hidden_evaluations": float(nevals.float().mean().item()),
        "newton_max_iterations": float(nits.max().item()),
        "time_ratio_newton_over_algebraic": float(t_newton / max(t_alg, 1e-9)),
        "inverse_contraction_inf": info["kappa_inf"],
        "Dhat": Dhat.numpy().tolist(),
    }

    ue, uaa, udd = u[te].numpy(), ua.numpy(), ud.numpy()
    fig, ax = plt.subplots(1, 3, figsize=(10.1, 3.0))
    ax[0].scatter(ue[:, 0], uaa[:, 0], s=4, alpha=0.35, label="algebraic")
    ax[0].scatter(ue[:, 0], udd[:, 0], s=4, alpha=0.22, label="direct inverse")
    lim = [-2.1, 2.1]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlim=lim, ylim=lim, xlabel=r"true $u_1$", ylabel=r"estimated $u_1$")
    ax[0].legend(frameon=False, fontsize=8)
    ea = np.linalg.norm(uaa - ue, axis=1)
    ed = np.linalg.norm(udd - ue, axis=1)
    bins = np.linspace(0, np.quantile(np.r_[ea, ed], 0.995), 42)
    ax[1].hist(ea, bins=bins, density=True, alpha=0.62, label="algebraic")
    ax[1].hist(ed, bins=bins, density=True, alpha=0.50, label="direct inverse")
    ax[1].set(xlabel=r"$\|\widehat u-u\|_2$", ylabel="density")
    ax[1].legend(frameon=False, fontsize=8)
    vals, counts = np.unique(info["iterations"].numpy(), return_counts=True)
    ax[2].bar(vals, counts / counts.sum(), width=0.8)
    ax[2].set(xlabel="fixed-point iterations", ylabel="fraction of points")
    for a in ax:
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "static_inverse_results.pdf", bbox_inches="tight")
    fig.savefig(FIG / "static_inverse_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    torch.save({"model": model.state_dict(), "inverse": inv.state_dict()}, RES / "static_models.pt")
    return out


# -----------------------------------------------------------------------------
# Experiment 2: square input-output inversion of a nonlinear NARX plant
# -----------------------------------------------------------------------------
M_IO = 2
NU_Y = 3
NU_U = 2
N_CHI = M_IO * (NU_Y + NU_U)

D_TRUE = np.array([[0.82, 0.07], [0.04, 0.76]], dtype=float)
A_Y1 = np.array([[0.46, 0.07], [-0.04, 0.43]], dtype=float)
A_Y2 = np.array([[-0.12, 0.025], [0.018, -0.10]], dtype=float)
A_Y3 = np.array([[0.035, -0.006], [0.004, 0.030]], dtype=float)
B_U1 = np.array([[0.070, 0.012], [-0.008, 0.055]], dtype=float)
B_U2 = np.array([[-0.020, 0.004], [0.006, -0.018]], dtype=float)
D_CHI_TRUE = np.hstack([A_Y1, A_Y2, A_Y3, B_U1, B_U2])

MEAS_NOISE_STD = 0.012
PROC_NOISE_BOUND = 0.003


def true_narx(chi: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    Dq = torch.as_tensor(D_CHI_TRUE, dtype=chi.dtype, device=chi.device)
    Du = torch.as_tensor(D_TRUE, dtype=chi.dtype, device=chi.device)
    lin = chi @ Dq.T + u @ Du.T
    y0 = chi[:, 0:2]
    ym1 = chi[:, 2:4]
    n1 = (
        0.085 * torch.sin(0.90 * y0[:, 0] + 0.25 * ym1[:, 1])
        + 0.035 * torch.tanh(u[:, 0] - 0.55 * u[:, 1])
        + 0.018 * torch.tanh(y0[:, 0] * ym1[:, 0])
    )
    n2 = (
        0.075 * torch.sin(0.35 * ym1[:, 0] + 0.90 * y0[:, 1])
        + 0.032 * torch.tanh(0.45 * u[:, 0] + u[:, 1])
        + 0.016 * torch.tanh(y0[:, 1] * ym1[:, 1])
    )
    return lin + torch.stack([n1, n2], 1)


def narx_shift(chi: torch.Tensor, y_new: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    # chi = col(y_k,y_{k-1},y_{k-2},u_{k-1},u_{k-2}).
    return torch.cat([y_new, chi[:, 0:4], u, chi[:, 6:8]], dim=1)


def narx_shift_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    J = np.zeros((N_CHI, N_CHI))
    G_y = np.zeros((N_CHI, M_IO))
    G_u = np.zeros((N_CHI, M_IO))
    P_y = np.zeros((M_IO, N_CHI))
    P_y[:, :M_IO] = np.eye(M_IO)
    G_y[:M_IO, :] = np.eye(M_IO)
    J[M_IO : NU_Y * M_IO, : (NU_Y - 1) * M_IO] = np.eye((NU_Y - 1) * M_IO)
    input_start = NU_Y * M_IO
    G_u[input_start : input_start + M_IO, :] = np.eye(M_IO)
    J[input_start + M_IO :, input_start : input_start + (NU_U - 1) * M_IO] = np.eye((NU_U - 1) * M_IO)
    return J, G_y, G_u, P_y


def generate_narx_data(ntraj: int, horizon: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Open-loop excitation: filtered piecewise-constant random targets; the
    regressor is built from *measured* outputs (noise std MEAS_NOISE_STD)."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    chi_data: list[torch.Tensor] = []
    u_data: list[torch.Tensor] = []
    y_data: list[torch.Tensor] = []
    for _ in range(ntraj):
        chi_true = torch.zeros(1, N_CHI, dtype=DTYPE)
        chi_meas = torch.zeros(1, N_CHI, dtype=DTYPE)
        u_filt = np.zeros(M_IO)
        target = rng.uniform(-1.2, 1.2, M_IO)
        for k in range(horizon):
            if k % 4 == 0:
                target = rng.uniform(-1.2, 1.2, M_IO)
            u_filt = 0.65 * u_filt + 0.35 * target
            u = torch.tensor(u_filt[None], dtype=DTYPE)
            y_true = true_narx(chi_true, u) + PROC_NOISE_BOUND * (2 * torch.rand(1, M_IO, generator=gen) - 1)
            y_meas = y_true + MEAS_NOISE_STD * torch.randn(1, M_IO, generator=gen)
            chi_data.append(chi_meas.clone())
            u_data.append(u.clone())
            y_data.append(y_meas.clone())
            chi_true = narx_shift(chi_true, y_true, u)
            chi_meas = narx_shift(chi_meas, y_meas, u)
    return torch.cat(chi_data), torch.cat(u_data), torch.cat(y_data)


def train_narx_model(chi, u, y, tr, D_u_hat, D_chi_hat, c_hat, *, seed, hdim=48, kappa_i=0.68,
                     sens_w=5e-4, input_features=True, kappa_f=0.0, steps=1000):
    torch.manual_seed(seed)
    model = AlgebraicImplicitMap(N_CHI, M_IO, hdim, D_u_hat, D_chi=D_chi_hat, kappa_i=kappa_i,
                                 input_features=input_features, kappa_f=kappa_f)
    model.c.data.copy_(c_hat)
    fit_forward(model, chi[tr], u[tr], y[tr], steps=steps, batch=1536, lr=2.2e-3, sensitivity_weight=sens_w)
    return model


def conformal_box(model: AlgebraicImplicitMap, chi_cal, u_cal, y_cal, ntraj: int, horizon: int, sigma: np.ndarray) -> dict:
    """Trajectory-wise conformal model-error envelope.

    Score of calibration trajectory j:  s_j = max_k max_i |r_{k,i}^{(j)}| / sigma_i,
    with a fixed scale sigma computed from the *training* trajectories.  The
    box  d_bar = (max_j s_j) sigma  contains the whole residual trajectory of a
    new exchangeable trajectory with probability >= 1 - 1/(ntraj+1)."""
    with torch.no_grad():
        r = (y_cal - model(chi_cal, u_cal)).numpy().reshape(ntraj, horizon, M_IO)
    per_traj_max = np.max(np.abs(r), axis=1)  # (ntraj, m)
    scores = np.max(per_traj_max / sigma[None, :], axis=1)
    q = float(np.max(scores))
    return {
        "d_bar": (q * sigma).tolist(),
        "sigma": sigma.tolist(),
        "quantile_score": q,
        "coverage_guarantee": 1.0 - 1.0 / (ntraj + 1),
        "per_sample_q999_box": (np.quantile(np.abs(r).reshape(-1, M_IO), 0.999, axis=0)).tolist(),
    }


def empirical_coverage(model, chi_te, u_te, y_te, ntraj, horizon, d_bar) -> float:
    with torch.no_grad():
        r = (y_te - model(chi_te, u_te)).numpy().reshape(ntraj, horizon, M_IO)
    inside = np.all(np.abs(r) <= np.asarray(d_bar)[None, None, :], axis=(1, 2))
    return float(np.mean(inside))


def certificate(sens: dict, F, E_chi, E_u, d_bar, w_bound, r_fp_bar) -> dict:
    J, G_y, G_u, P_y = narx_shift_matrices()
    S_chi, S_y, K_u, K_y = sens["S_chi"], sens["S_y"], sens["K_u"], sens["K_y"]
    L_u = S_chi + S_y @ np.abs(F) @ P_y
    L_y = np.abs(F) @ P_y + E_chi + E_u @ L_u
    H = J + G_y @ L_y + G_u @ L_u
    G_fp = G_y @ (K_y + E_u @ K_u) + G_u @ K_u
    rho_H = pf_radius(H)
    out = {"L_u": L_u, "L_y": L_y, "H": H, "rho_H": rho_H,
           "rho_nominal_memory": pf_radius(J + G_u @ S_chi),
           "rho_no_uncertainty": pf_radius(J + G_y @ (np.abs(F) @ P_y) + G_u @ L_u)}
    if rho_H < 1:
        g = G_fp @ r_fp_bar + G_y @ (np.asarray(d_bar) + w_bound)
        chi_tube = np.linalg.solve(np.eye(N_CHI) - H, g)
        out.update({"chi_tube": chi_tube, "y_tube": P_y @ chi_tube, "u_tube": L_u @ chi_tube + K_u @ r_fp_bar,
                    "g_bar": g})
    return out


def slope_bounds(model: AlgebraicImplicitMap, z_ref: np.ndarray, chi_box: np.ndarray, F, lam_prev: np.ndarray) -> np.ndarray:
    """Propose slopes from inverse-solution intervals. These are diagnostic:
    forward equilibria and all solver iterates need separate containment
    checks before slopes below one can be used in the paper's certificates.
    """
    sens = model.sensitivity_matrices(lam_prev)
    _, _, _, P_y = narx_shift_matrices()
    Lam = np.diag(lam_prev)
    dv = np.abs(F) @ P_y @ chi_box  # deviation of the requested output
    rhs_dev = np.abs(sens["B_i_chi"]) @ chi_box + np.abs(sens["B_i_y"]) @ dv
    xi_dev = sens["R_i"] @ Lam @ rhs_dev
    dz = np.abs(sens["A_i"]) @ xi_dev + rhs_dev + 1e-6
    lo = z_ref.min(axis=0) - dz
    hi = z_ref.max(axis=0) + dz
    dist = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(np.abs(lo), np.abs(hi)))
    lam = 1.0 / np.cosh(dist) ** 2
    return np.minimum(lam, lam_prev)


def build_reference(model: AlgebraicImplicitMap, T: int):
    kk = np.arange(T + 1)
    ramp = 1.0 - np.exp(-kk / 18.0)
    y_ref = np.column_stack(
        [
            ramp * (0.55 * np.sin(2 * np.pi * kk / 100) + 0.16 * np.sin(2 * np.pi * kk / 37)),
            ramp * (0.48 * np.sin(2 * np.pi * kk / 130 + 0.4) - 0.13 * np.sin(2 * np.pi * kk / 43)),
        ]
    )
    chi_ref = np.zeros((T + 1, N_CHI))
    u_ref = np.zeros((T, M_IO))
    z_ref = np.zeros((T, model.hdim))
    warm = None
    for k in range(T):
        qk = torch.tensor(chi_ref[k : k + 1], dtype=DTYPE)
        yk1 = torch.tensor(y_ref[k + 1 : k + 2], dtype=DTYPE)
        uk, inf = model.inverse(qk, yk1, tol=1e-8, warm=warm, return_info=True)
        warm = inf["h"]
        u_ref[k] = uk.numpy()[0]
        z_ref[k] = inf["z"].numpy()[0]
        chi_ref[k + 1] = narx_shift(qk, yk1, uk).numpy()[0]
    return y_ref, chi_ref, u_ref, z_ref


def closed_loop_narx(controller: str, model, aux, y_ref, chi_ref, u_ref, F, E_chi, E_u, bias_bound, w_bound,
                     nruns=120, seed=SEED + 20, scale_uncertainty=1.0, burn=40, init_range=0.15):
    """Monte Carlo closed loop over the uncertain plant family (84).

    The controller only sees *measured* outputs (noise std MEAS_NOISE_STD) and
    builds its regressor q from them; the plant evolves on its true history.
    Tracking metrics are reported for the true output; the tube check uses the
    measured error, which is the quantity bounded by the theorem."""
    T = len(u_ref)
    rng = np.random.default_rng(seed)
    delta_chi = rng.uniform(-1, 1, (nruns, M_IO, N_CHI)) * E_chi[None] * scale_uncertainty
    delta_u = rng.uniform(-1, 1, (nruns, M_IO, M_IO)) * E_u[None] * scale_uncertainty
    run_bias = rng.uniform(-1, 1, (nruns, M_IO)) * bias_bound[None] * scale_uncertainty
    process_noise = rng.uniform(-1, 1, (nruns, T, M_IO)) * w_bound[None, None]
    meas_noise = MEAS_NOISE_STD * rng.standard_normal((nruns, T, M_IO))
    initial_y = rng.uniform(-init_range, init_range, (nruns, M_IO))
    q = np.zeros((nruns, N_CHI))       # measured regressor (controller)
    q[:, 0:2] = initial_y
    q[:, 2:4] = initial_y
    q_true = q.copy()                  # true regressor (plant)
    warm = None
    u_prev = torch.zeros(nruns, M_IO)
    y_hist = np.zeros((nruns, T, M_IO))
    ym_hist = np.zeros((nruns, T, M_IO))
    u_hist = np.zeros((nruns, T, M_IO))
    q_hist = np.zeros((nruns, T, N_CHI))
    fp_max = 0.0
    iter_hist, eval_hist = [], []
    t_ctrl = 0.0
    for k in range(T):
        e = q[:, :M_IO] - y_ref[k]
        v = y_ref[k + 1][None, :] + e @ F.T
        qt = torch.tensor(q, dtype=DTYPE)
        q_hist[:, k] = q
        vt = torch.tensor(v, dtype=DTYPE)
        t0 = time.perf_counter()
        if controller == "implicit inverse":
            uc, inf = model.inverse(qt, vt, tol=2e-7, warm=warm, return_info=True)
            warm = inf["h"]
            uk = uc.numpy()
            iter_hist.append(inf["iterations"].numpy())
            eval_hist.append(inf["iterations"].numpy())
            fp_max = max(fp_max, float(inf["residual"].max()))
        elif controller == "online Newton":
            uc, nits, nev = newton_inverse(model, qt, vt, u_prev, tol=2e-7)
            u_prev = uc
            uk = uc.numpy()
            iter_hist.append(nits.numpy())
            eval_hist.append(nev.numpy())
        elif controller == "direct inverse":
            with torch.no_grad():
                uk = aux["direct"](torch.cat([qt, vt], 1)).numpy()
        elif controller == "linear ARX inverse":
            uk = ((vt - qt @ aux["D_chi_hat"].T - aux["c_hat"]) @ torch.linalg.inv(aux["D_u_hat"]).T).numpy()
        else:
            raise ValueError(controller)
        t_ctrl += time.perf_counter() - t0
        uk = np.clip(uk, -2.2, 2.2)
        y_nom = true_narx(torch.tensor(q_true, dtype=DTYPE), torch.tensor(uk, dtype=DTYPE)).numpy()
        q_dev = q - chi_ref[k]
        u_dev = uk - u_ref[k]
        y_next = (
            y_nom
            + np.einsum("nij,nj->ni", delta_chi, q_dev)
            + np.einsum("nij,nj->ni", delta_u, u_dev)
            + run_bias
            + process_noise[:, k]
        )
        y_meas = y_next + meas_noise[:, k]
        y_hist[:, k] = y_next
        ym_hist[:, k] = y_meas
        u_hist[:, k] = uk
        q = narx_shift(qt, torch.tensor(y_meas, dtype=DTYPE), torch.tensor(uk, dtype=DTYPE)).numpy()
        q_true = narx_shift(torch.tensor(q_true, dtype=DTYPE), torch.tensor(y_next, dtype=DTYPE), torch.tensor(uk, dtype=DTYPE)).numpy()

    err = y_hist - y_ref[1:][None]
    err_m = ym_hist - y_ref[1:][None]
    rr = np.sqrt(np.mean(err**2, axis=(1, 2)))
    mm = np.max(np.linalg.norm(err, axis=2), axis=1)
    uu = np.sqrt(np.mean(u_hist**2, axis=(1, 2)))
    ss = np.mean(np.abs(u_hist) >= 2.199, axis=(1, 2))
    metrics = {
        "rmse_mean": float(np.mean(rr)),
        "rmse_p95": float(np.quantile(rr, 0.95)),
        "max_error_mean": float(np.mean(mm)),
        "max_error_p95": float(np.quantile(mm, 0.95)),
        "rms_input_mean": float(np.mean(uu)),
        "saturation_fraction_mean": float(np.mean(ss)),
        "controller_time_per_step_ms": float(1e3 * t_ctrl / T),
        "rmse_samples": rr.tolist(),
        "max_error_samples": mm.tolist(),
    }
    if iter_hist:
        all_it = np.concatenate(iter_hist)
        metrics["closed_loop_iterations_mean"] = float(np.mean(all_it))
        metrics["closed_loop_iterations_p95"] = float(np.quantile(all_it, 0.95))
        metrics["closed_loop_iterations_max"] = float(np.max(all_it))
    if eval_hist:
        all_ev = np.concatenate(eval_hist)
        metrics["hidden_evaluations_mean"] = float(np.mean(all_ev))
    return metrics, {"y": y_hist, "ym": ym_hist, "u": u_hist, "chi": q_hist,
                     "fp_residual_max": fp_max, "err_m": err_m, "u_dev": u_hist - u_ref[None]}


def calibrate_closed_loop(model, aux, y_ref, chi_ref, u_ref, F, E_chi, E_u,
                          bias_bound, w_bound, sigma, preliminary_box,
                          ncal=20, seed=SEED + 2026):
    """Fresh calibration after freezing model, controller and reference.

    The score bounds the FULL measured prediction residual. The controller
    does not use this envelope, so recomputing its certificate does not change
    the trajectory law. The older selection box is retained as a conservative
    floor; this preserves coverage for the independent calibration draw.
    """
    _, traj = closed_loop_narx('implicit inverse', model, aux, y_ref, chi_ref,
                              u_ref, F, E_chi, E_u, bias_bound, w_bound,
                              nruns=ncal, seed=seed)
    T = len(u_ref)
    with torch.no_grad():
        pred = model(torch.tensor(traj['chi'].reshape(-1, N_CHI), dtype=DTYPE),
                     torch.tensor(traj['u'].reshape(-1, M_IO), dtype=DTYPE)).numpy()
    residual = traj['ym'] - pred.reshape(ncal, T, M_IO)
    sigma = np.asarray(sigma)
    scores = np.max(np.abs(residual) / sigma[None, None, :], axis=(1, 2))
    fresh_box = scores.max() * sigma
    return {'seed': seed, 'trajectories': ncal, 'horizon': T,
            'sigma': sigma.tolist(), 'scores': scores.tolist(),
            'fresh_residual_box': fresh_box.tolist(),
            'final_residual_box': np.maximum(fresh_box, preliminary_box).tolist(),
            'coverage_guarantee': 1 - 1 / (ncal + 1),
            'fp_residual_max': traj['fp_residual_max']}


def tube_check(traj: dict, y_tube, u_tube, burn=40) -> dict:
    y_ratio = np.abs(traj["err_m"][:, burn:, :]) / np.asarray(y_tube)[None, None, :]
    u_ratio = np.abs(traj["u_dev"][:, burn:, :]) / np.asarray(u_tube)[None, None, :]
    return {
        "output_tube_exceedance_fraction_after_burn": float(np.mean(y_ratio > 1.0)),
        "input_tube_exceedance_fraction_after_burn": float(np.mean(u_ratio > 1.0)),
        "output_tube_max_ratio_after_burn": float(np.max(y_ratio)),
        "input_tube_max_ratio_after_burn": float(np.max(u_ratio)),
    }


def run_narx() -> dict:
    ntraj, horizon = 180, 110
    chi, u, y = generate_narx_data(ntraj=ntraj, horizon=horizon, seed=SEED + 2)
    T = 280
    n_cal_traj = 20
    chi_c, u_c, y_c = generate_narx_data(ntraj=n_cal_traj, horizon=T, seed=SEED + 3)

    # Split the identification data by complete trajectories.
    split_rng = np.random.default_rng(SEED + 7)
    traj_order = split_rng.permutation(ntraj)
    n_train_traj = 150
    train_traj = traj_order[:n_train_traj]
    test_traj = traj_order[n_train_traj:]

    def sample_indices(traj_ids: np.ndarray) -> torch.Tensor:
        idx = np.concatenate([np.arange(int(j) * horizon, (int(j) + 1) * horizon) for j in traj_ids])
        return torch.as_tensor(idx, dtype=torch.long)

    tr = sample_indices(train_traj)
    te = sample_indices(test_traj)

    # Linear ARX fit provides the linear skip and the linear baseline.
    Xaug = torch.cat([chi[tr], u[tr], torch.ones(len(tr), 1)], 1)
    coef = torch.linalg.lstsq(Xaug, y[tr]).solution
    D_chi_hat = coef[:N_CHI].T.detach()
    D_u_hat = coef[N_CHI : N_CHI + M_IO].T.detach()
    c_hat = coef[-1].detach()
    aux = {"D_chi_hat": D_chi_hat, "D_u_hat": D_u_hat, "c_hat": c_hat}

    # ---- models: main + ablations -------------------------------------------
    LS = 1e-4  # inverse-sensitivity penalty weight of the main model
    variants = {
        "main": dict(sens_w=LS),
        "control-affine (B_u=0)": dict(sens_w=LS, input_features=False),
        "control-affine (B_u=0), lambda_s=3e-3": dict(sens_w=3e-3, input_features=False),
        "lambda_s=0": dict(sens_w=0.0),
        "lambda_s=1e-3": dict(sens_w=1e-3),
        "implicit forward (A!=0)": dict(sens_w=LS, kappa_f=0.35),
    }
    models = {}
    for name, kw in variants.items():
        print(f"    training variant: {name}", flush=True)
        models[name] = train_narx_model(chi, u, y, tr, D_u_hat, D_chi_hat, c_hat, seed=SEED + 2, **kw)
    model = models["main"]
    torch.manual_seed(SEED + 2)
    direct = StandardizedMLP(torch.cat([chi[tr], y[tr]], 1), u[tr], width=96, depth=3)
    fit_mlp(direct, torch.cat([chi[tr], y[tr]], 1), u[tr], steps=1000, batch=1536, lr=2.0e-3)
    aux["direct"] = direct

    # ---- certificate ingredients ---------------------------------------------
    F = 0.10 * np.eye(M_IO)
    E_chi = np.array(
        [
            [0.0040, 0.0020, 0.0020, 0.0010, 0.0010, 0.0010, 0.0010, 0.0005, 0.0005, 0.0005],
            [0.0020, 0.0040, 0.0010, 0.0020, 0.0010, 0.0010, 0.0005, 0.0010, 0.0005, 0.0005],
        ]
    )
    E_u = np.array([[0.0040, 0.0015], [0.0010, 0.0040]])
    bias_bound = np.array([0.004, 0.004])
    w_bound = np.array([PROC_NOISE_BOUND, PROC_NOISE_BOUND])
    J, G_y, G_u, P_y = narx_shift_matrices()

    # ---- open-loop tests, per variant ----------------------------------------
    with torch.no_grad():
        y_clean = true_narx(chi[te], u[te])
        u_direct = direct(torch.cat([chi[te], y_clean], 1))
        u_linear = (y_clean - chi[te] @ D_chi_hat.T - c_hat) @ torch.linalg.inv(D_u_hat).T
    def training_scale(mdl):
        with torch.no_grad():
            r = (y[tr] - mdl(chi[tr], u[tr])).numpy().reshape(n_train_traj, horizon, M_IO)
        return np.median(np.max(np.abs(r), axis=1), axis=0)

    variant_results = {}
    for name, mdl in models.items():
        with torch.no_grad():
            pred = mdl(chi[te], u[te])
            u_alg, info = mdl.inverse(chi[te], y_clean, return_info=True)
            cycle = mdl(chi[te], u_alg)
        sens = mdl.sensitivity_matrices()
        r_fp_bar = 2e-7 * np.ones(mdl.hdim)
        box = conformal_box(mdl, chi_c, u_c, y_c, n_cal_traj, T, training_scale(mdl))
        d_bar = np.asarray(box["d_bar"]) + bias_bound
        cert = certificate(sens, F, E_chi, E_u, d_bar, w_bound, r_fp_bar)
        variant_results[name] = {
            "forward_rmse": rms((pred - y_clean).numpy()),
            "inverse_rmse_algebraic": rms((u_alg - u[te]).numpy()),
            "learned_model_cycle_rmse": rms((cycle - y_clean).numpy()),
            "inverse_contraction_inf": info["kappa_inf"],
            "rho_absA": sens["rho_absA"],
            "rho_absAi": sens["rho_absAi"],
            "mean_iterations": float(info["iterations"].float().mean().item()),
            "p95_iterations": float(torch.quantile(info["iterations"].float(), 0.95).item()),
            "S_chi_inf": float(np.max(np.sum(sens["S_chi"], axis=1))),
            "rho_H": cert["rho_H"],
            "rho_nominal_memory": cert["rho_nominal_memory"],
            "conformal_box": box["d_bar"],
            "y_tube": cert.get("y_tube", np.full(M_IO, np.nan)).tolist(),
            "u_tube": cert.get("u_tube", np.full(M_IO, np.nan)).tolist(),
        }
        print(f"    {name}: fwd {variant_results[name]['forward_rmse']:.5f}  inv {variant_results[name]['inverse_rmse_algebraic']:.5f}  rho_H {cert['rho_H']:.4f}", flush=True)

    # ---- main model: certificate, conformal box, slope refinement ------------
    sens = model.sensitivity_matrices()
    r_fp_bar = 2e-7 * np.ones(model.hdim)
    box = conformal_box(model, chi_c, u_c, y_c, n_cal_traj, T, training_scale(model))
    coverage_test = empirical_coverage(model, chi[te], u[te], y[te], len(test_traj), horizon, box["d_bar"])
    d_bar = np.asarray(box["d_bar"]) + bias_bound
    cert0 = certificate(sens, F, E_chi, E_u, d_bar, w_bound, r_fp_bar)
    # signed vs majorant zero dynamics of the fitted linear part
    K_lin = np.linalg.solve(D_u_hat.numpy().astype(float), D_chi_hat.numpy().astype(float))
    rho_signed_linear = pf_radius(J - G_u @ K_lin)
    rho_majorant_linear = pf_radius(J + G_u @ np.abs(K_lin))

    y_ref, chi_ref, u_ref, z_ref = build_reference(model, T)
    final_calibration = calibrate_closed_loop(
        model, aux, y_ref, chi_ref, u_ref, F, E_chi, E_u, bias_bound, w_bound,
        box['sigma'], box['d_bar'])
    d_bar = np.asarray(final_calibration['final_residual_box']) + bias_bound
    cert0 = certificate(sens, F, E_chi, E_u, d_bar, w_bound, r_fp_bar)
    variant_results['main']['y_tube'] = cert0['y_tube'].tolist()
    variant_results['main']['u_tube'] = cert0['u_tube'].tolist()
    init_range = 0.15
    # invariant box containing the initial regressor error: t * chi_tube, t >= 1
    chi0_bound = np.zeros(N_CHI)
    chi0_bound[0:4] = init_range
    t_scale = max(1.0, float(np.max(chi0_bound / cert0["chi_tube"])))
    chi_box0 = t_scale * cert0["chi_tube"]
    zero_cross = float(np.mean((z_ref.min(axis=0) <= 0) & (z_ref.max(axis=0) >= 0)))
    lam = np.ones(model.hdim)
    lam_hist = []
    cert = cert0
    for it in range(6):
        lam_new = slope_bounds(model, z_ref, chi_box0, F, lam)
        # Retain the global bound unless ALL intervals required by Remark 1
        # have been verified, including forward solutions and solver iterates.
        # The supplied benchmark uses Lambda = I.
        lam_new = np.ones_like(lam_new)
        sens_l = model.sensitivity_matrices(lam_new)
        cert_new = certificate(sens_l, F, E_chi, E_u, d_bar, w_bound, r_fp_bar)
        lam_hist.append({"iteration": it + 1, "rho_H": cert_new["rho_H"], "lam_mean": float(lam_new.mean()),
                         "lam_max": float(lam_new.max()), "lam_min": float(lam_new.min())})
        if np.max(np.abs(lam_new - lam)) < 1e-4:
            lam, cert, sens_lam = lam_new, cert_new, sens_l
            break
        lam, cert, sens_lam = lam_new, cert_new, sens_l
    # the refined box t*chi_tube_Lambda-based restart is not needed: t*chi_tube0
    # remains invariant for the refined recursion (H_Lambda <= H componentwise).

    # ---- closed-loop Monte Carlo ----------------------------------------------
    controllers = ["implicit inverse", "online Newton", "direct inverse", "linear ARX inverse"]
    metrics, trajs = {}, {}
    for ctrl in controllers:
        print(f"    closed loop: {ctrl}", flush=True)
        metrics[ctrl], trajs[ctrl] = closed_loop_narx(ctrl, model, aux, y_ref, chi_ref, u_ref, F, E_chi, E_u, bias_bound, w_bound)
    metrics["implicit inverse"].update({"tube_plain_" + k: v for k, v in tube_check(trajs["implicit inverse"], cert0["y_tube"], cert0["u_tube"]).items()})
    metrics["implicit inverse"].update({"tube_slope_" + k: v for k, v in tube_check(trajs["implicit inverse"], cert["y_tube"], cert["u_tube"]).items()})
    # ablation models in closed loop (their own inverse)
    for name in [n for n in variants if n != "main"]:
        print(f"    closed loop: {name}", flush=True)
        yr, cr, ur, _ = build_reference(models[name], T)
        m_, _ = closed_loop_narx("implicit inverse", models[name], aux, yr, cr, ur, F, E_chi, E_u, bias_bound, w_bound)
        variant_results[name]["closed_loop"] = {k: v for k, v in m_.items() if k not in ("rmse_samples", "max_error_samples")}
        variant_results[name]["closed_loop_rmse_samples"] = m_["rmse_samples"]
        if name == "implicit forward (A!=0)":
            mn_, _ = closed_loop_narx("online Newton", models[name], aux, yr, cr, ur, F, E_chi, E_u, bias_bound, w_bound)
            variant_results[name]["closed_loop_newton"] = {k: v for k, v in mn_.items() if k not in ("rmse_samples", "max_error_samples")}
    variant_results["main"]["closed_loop"] = {k: v for k, v in metrics["implicit inverse"].items() if k not in ("rmse_samples", "max_error_samples")}
    # plants violating the envelope
    print("    closed loop: envelope violation batch", flush=True)
    viol = {}
    for sc in [1.5, 2.0]:
        m_, tr_ = closed_loop_narx("implicit inverse", model, aux, y_ref, chi_ref, u_ref, F, E_chi, E_u, bias_bound, w_bound, scale_uncertainty=sc, seed=SEED + 21)
        viol[f"scale_{sc}"] = {k: v for k, v in m_.items() if k not in ("rmse_samples", "max_error_samples")}
        viol[f"scale_{sc}"].update(tube_check(tr_, cert["y_tube"], cert["u_tube"]))
    # paired comparisons
    paired = {}
    for other in ["online Newton", "direct inverse", "linear ARX inverse"]:
        paired[f"implicit minus {other}"] = paired_bootstrap_ci(metrics["implicit inverse"]["rmse_samples"], metrics[other]["rmse_samples"])
    paired["implicit minus control-affine"] = paired_bootstrap_ci(metrics["implicit inverse"]["rmse_samples"], variant_results["control-affine (B_u=0)"]["closed_loop_rmse_samples"])

    out = {
        "input_dimension": M_IO,
        "output_dimension": M_IO,
        "regressor_dimension": N_CHI,
        "output_history": NU_Y,
        "input_history": NU_U,
        "training_trajectories": int(n_train_traj),
        "test_trajectories": int(len(test_traj)),
        "calibration_trajectories": int(n_cal_traj),
        "calibration_horizon": T,
        "closed_loop_horizon": T,
        "measurement_noise_std": MEAS_NOISE_STD,
        "forward_rmse": variant_results["main"]["forward_rmse"],
        "inverse_rmse_algebraic": variant_results["main"]["inverse_rmse_algebraic"],
        "inverse_rmse_direct": rms((u_direct - u[te]).numpy()),
        "inverse_rmse_linear": rms((u_linear - u[te]).numpy()),
        "learned_model_cycle_rmse": variant_results["main"]["learned_model_cycle_rmse"],
        "inverse_contraction_inf": variant_results["main"]["inverse_contraction_inf"],
        "mean_iterations": variant_results["main"]["mean_iterations"],
        "p95_iterations": variant_results["main"]["p95_iterations"],
        "D_u_hat": D_u_hat.numpy().tolist(),
        "E_chi": E_chi.tolist(),
        "E_u": E_u.tolist(),
        "F": F.tolist(),
        "conformal": box,
        "closed_loop_conformal": final_calibration,
        "conformal_empirical_coverage_test_trajectories": coverage_test,
        "d_bar_total": d_bar.tolist(),
        "S_chi": sens["S_chi"].tolist(),
        "S_y": sens["S_y"].tolist(),
        "L_u": cert0["L_u"].tolist(),
        "H": cert0["H"].tolist(),
        "rho_H": cert0["rho_H"],
        "rho_nominal_memory": cert0["rho_nominal_memory"],
        "rho_no_uncertainty": cert0["rho_no_uncertainty"],
        "rho_signed_linear_zero_dynamics": rho_signed_linear,
        "rho_majorant_linear_zero_dynamics": rho_majorant_linear,
        "output_tube": cert0["y_tube"].tolist(),
        "input_deviation_tube": cert0["u_tube"].tolist(),
        "slope_refinement": {
            "invariant_box_scale_t": t_scale,
            "fraction_neurons_reference_preactivation_crosses_zero": zero_cross,
            "preactivation_abs_max": float(np.max(np.abs(z_ref))),
            "iterations": lam_hist,
            "lam": lam.tolist(),
            "lam_mean": float(lam.mean()),
            "lam_max": float(lam.max()),
            "lam_min": float(lam.min()),
            "rho_H": cert["rho_H"],
            "rho_nominal_memory": cert["rho_nominal_memory"],
            "S_chi_inf": float(np.max(np.sum(sens_lam["S_chi"], axis=1))),
            "output_tube": cert["y_tube"].tolist(),
            "input_deviation_tube": cert["u_tube"].tolist(),
        },
        "reference_input_max": float(np.max(np.abs(u_ref))),
        "variants": variant_results,
        "envelope_violation": viol,
        "paired_rmse_differences": paired,
        "monte_carlo": metrics,
    }

    # ---- figures ----------------------------------------------------------------
    tt = np.arange(T)
    fig, ax = plt.subplots(2, 2, figsize=(3.6, 3.1), sharex="col")
    plot_ctrls = ["implicit inverse", "direct inverse", "linear ARX inverse"]
    labels = {"implicit inverse": "implicit inv.", "direct inverse": "direct inv.", "linear ARX inverse": "linear ARX"}
    for ctrl in plot_ctrls:
        ys, us = trajs[ctrl]["y"][0], trajs[ctrl]["u"][0]
        ax[0, 0].plot(tt + 1, ys[:, 0], lw=0.8, label=labels[ctrl])
        ax[0, 1].plot(tt + 1, ys[:, 1], lw=0.8)
        ax[1, 0].plot(tt, us[:, 0], lw=0.7)
        ax[1, 1].plot(tt, us[:, 1], lw=0.7)
    yt, ut = cert["y_tube"], cert["u_tube"]
    ax[0, 0].fill_between(tt + 1, y_ref[1:, 0] - yt[0], y_ref[1:, 0] + yt[0], alpha=0.15, label="tube")
    ax[0, 1].fill_between(tt + 1, y_ref[1:, 1] - yt[1], y_ref[1:, 1] + yt[1], alpha=0.15)
    ax[1, 0].fill_between(tt, u_ref[:, 0] - ut[0], u_ref[:, 0] + ut[0], alpha=0.15)
    ax[1, 1].fill_between(tt, u_ref[:, 1] - ut[1], u_ref[:, 1] + ut[1], alpha=0.15)
    ax[0, 0].plot(tt + 1, y_ref[1:, 0], "k--", lw=0.9, label="reference")
    ax[0, 1].plot(tt + 1, y_ref[1:, 1], "k--", lw=0.9)
    for a, lab in zip(ax.flat, [r"$y_1$", r"$y_2$", r"$u_1$", r"$u_2$"]):
        a.set_ylabel(lab, fontsize=7, labelpad=1)
        a.tick_params(labelsize=6)
        a.grid(True, alpha=0.25)
    ax[1, 0].set_xlabel("sample", fontsize=7)
    ax[1, 1].set_xlabel("sample", fontsize=7)
    handles, labels_ = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.03), columnspacing=0.9, handlelength=1.4)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.965))
    fig.savefig(FIG / "narx_tracking.pdf", bbox_inches="tight")
    fig.savefig(FIG / "narx_tracking.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
    names = ["implicit", "Newton", "direct", "linear"]
    ax[0].boxplot([metrics[m]["rmse_samples"] for m in controllers], tick_labels=names, showfliers=False)
    ax[1].boxplot([metrics[m]["max_error_samples"] for m in controllers], tick_labels=names, showfliers=False)
    ax[0].set_ylabel("tracking RMSE")
    ax[1].set_ylabel("maximum error norm")
    for a in ax:
        a.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "narx_monte_carlo.pdf", bbox_inches="tight")
    fig.savefig(FIG / "narx_monte_carlo.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    alphas = np.linspace(0.02, 0.99, 100)
    rhos, rhos_l, gains = [], [], []
    for alpha in alphas:
        Fa = alpha * np.eye(M_IO)
        c0 = certificate(sens, Fa, E_chi, E_u, d_bar, w_bound, r_fp_bar)
        c1 = certificate(sens_lam, Fa, E_chi, E_u, d_bar, w_bound, r_fp_bar)
        rhos.append(c0["rho_H"])
        rhos_l.append(c1["rho_H"])
        gains.append(np.max(np.sum(c0["L_u"], axis=1)))
    out["alpha_crossing_plain"] = float(alphas[np.argmax(np.array(rhos) >= 1.0)]) if np.any(np.array(rhos) >= 1) else None
    out["alpha_crossing_slope"] = float(alphas[np.argmax(np.array(rhos_l) >= 1.0)]) if np.any(np.array(rhos_l) >= 1) else None
    fig, ax1 = plt.subplots(figsize=(5.3, 3.15))
    l1 = ax1.plot(alphas, rhos, label=r"$\rho(\mathcal{H})$")
    ax1.axhline(1, color="k", ls="--", lw=1)
    ax1.set(xlabel=r"target pole $\alpha$ in $F=\alpha I$", ylabel=r"certificate $\rho(\mathcal{H})$")
    ax2 = ax1.twinx()
    l2 = ax2.plot(alphas, gains, ls=":", label=r"$\|L_u\|_\infty$")
    ax2.set_ylabel("inverse control-gain bound")
    ax1.grid(True, alpha=0.25)
    ax1.legend(l1 + l2, [a.get_label() for a in l1 + l2], frameon=False, fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(FIG / "certificate_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(FIG / "certificate_tradeoff.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    torch.save({"model": model.state_dict(), "inverse": direct.state_dict(),
                **{f"variant:{k}": v.state_dict() for k, v in models.items()}}, RES / "narx_models.pt")
    return out


# -----------------------------------------------------------------------------
# Experiment 3: physics-guided electromechanical servo
# -----------------------------------------------------------------------------
@dataclass
class ServoParams:
    J: float = 0.085
    kt: float = 0.58
    b: float = 0.14
    tc: float = 0.22
    cog: float = 0.075


def servo_step_np(theta, omega, voltage, p: ServoParams, h=0.02, load=0.0):
    acc = (
        p.kt * voltage
        - p.b * omega
        - p.tc * np.tanh(omega / 0.055)
        - p.cog * np.sin(6.0 * theta)
        - load
    ) / p.J
    omega_next = omega + h * acc
    theta_next = theta + h * omega_next
    return theta_next, omega_next


def make_servo_base(p: ServoParams, h: float):
    def base(chi: torch.Tensor):
        omega = chi[:, 1]
        return (omega + h * (-p.b * omega - p.tc * torch.tanh(omega / 0.055)) / p.J).unsqueeze(1)

    return base


def run_servo() -> dict:
    torch.manual_seed(SEED + 4)
    rng = np.random.default_rng(SEED + 4)
    ptrue = ServoParams()
    pnom = ServoParams(J=0.080, kt=0.55, b=0.115, tc=0.17, cog=0.0)
    h = 0.02
    D_u = np.array([[h * pnom.kt / pnom.J]])
    n = 8000
    theta = rng.uniform(-np.pi, np.pi, n)
    omega = rng.uniform(-5.0, 5.0, n)
    voltage = rng.uniform(-2.7, 2.7, n)
    omega_next = np.empty(n)
    for i in range(n):
        omega_next[i] = servo_step_np(theta[i], omega[i], voltage[i], ptrue, h)[1]
    omega_next_noisy = omega_next + rng.normal(0.0, 0.018, n)
    chi = torch.tensor(np.c_[theta, omega], dtype=DTYPE)
    u = torch.tensor(voltage[:, None], dtype=DTYPE)
    y0 = torch.tensor(omega_next[:, None], dtype=DTYPE)
    y = torch.tensor(omega_next_noisy[:, None], dtype=DTYPE)
    perm = torch.randperm(n)
    tr, te = perm[:6400], perm[6400:]
    torch.manual_seed(SEED + 4)
    model = AlgebraicImplicitMap(2, 1, 36, D_u, kappa_i=0.72, base_fn=make_servo_base(pnom, h))
    fit_forward(model, chi[tr], u[tr], y[tr], steps=1450, batch=1024, lr=2.7e-3, wd=3e-6)
    torch.manual_seed(SEED + 4)
    model_ca = AlgebraicImplicitMap(2, 1, 36, D_u, kappa_i=0.72, base_fn=make_servo_base(pnom, h), input_features=False)
    fit_forward(model_ca, chi[tr], u[tr], y[tr], steps=1450, batch=1024, lr=2.7e-3, wd=3e-6)
    torch.manual_seed(SEED + 4)
    direct = StandardizedMLP(torch.cat([chi[tr], y[tr]], 1), u[tr], width=56, depth=3)
    fit_mlp(direct, torch.cat([chi[tr], y[tr]], 1), u[tr], steps=1550, batch=1024, lr=2.5e-3)
    with torch.no_grad():
        pred = model(chi[te], u[te])
        pred_ca = model_ca(chi[te], u[te])
        ua, info = model.inverse(chi[te], y0[te], return_info=True)
        uca = model_ca.inverse(chi[te], y0[te])
        ud = direct(torch.cat([chi[te], y0[te]], 1))

    T = 700
    t_full = np.arange(T + 1) * h
    t = t_full[:-1]
    theta_ref = 0.70 * np.sin(0.62 * t_full) + 0.22 * np.sin(1.35 * t_full + 0.4)
    omega_ref = np.gradient(theta_ref, h)
    acc_ref = np.gradient(omega_ref, h)
    methods = ["implicit inverse", "control-affine inverse", "direct inverse", "nominal computed torque"]
    nruns = 80
    draws = []
    for _ in range(nruns):
        draws.append(
            ServoParams(
                J=ptrue.J * rng.uniform(0.90, 1.10),
                kt=ptrue.kt * rng.uniform(0.90, 1.10),
                b=ptrue.b * rng.uniform(0.80, 1.20),
                tc=ptrue.tc * rng.uniform(0.80, 1.20),
                cog=ptrue.cog * rng.uniform(0.75, 1.25),
            )
        )
    process_noise = 0.0015 * rng.standard_normal((nruns, T))
    meas_noise = 0.004 * rng.standard_normal((nruns, T + 1, 2))
    phases = rng.uniform(0, 2 * np.pi, nruns)
    metrics = {}
    trajectories = {}
    for method in methods:
        th = np.zeros((nruns, T + 1))
        om = np.zeros_like(th)
        vv = np.zeros((nruns, T))
        th[:, 0] = 0.18
        om[:, 0] = -0.08
        warm = None
        for k in range(T):
            th_m = th[:, k] + meas_noise[:, k, 0]
            om_m = om[:, k] + meas_noise[:, k, 1]
            a_des = acc_ref[k] - 18.0 * (th_m - theta_ref[k]) - 6.5 * (om_m - omega_ref[k])
            velocity_target = om_m + h * a_des
            q = torch.tensor(np.c_[th_m, om_m], dtype=DTYPE)
            target = torch.tensor(velocity_target[:, None], dtype=DTYPE)
            if method == "implicit inverse":
                ut, inf = model.inverse(q, target, tol=2e-7, warm=warm, return_info=True)
                warm = inf["h"]
                uk = ut.numpy()[:, 0]
            elif method == "control-affine inverse":
                uk = model_ca.inverse(q, target, tol=2e-7).numpy()[:, 0]
            elif method == "direct inverse":
                with torch.no_grad():
                    uk = direct(torch.cat([q, target], 1)).numpy()[:, 0]
            else:
                uk = (pnom.J * a_des + pnom.b * om_m + pnom.tc * np.tanh(om_m / 0.055)) / pnom.kt
            uk = np.clip(uk, -3.2, 3.2)
            vv[:, k] = uk
            for j in range(nruns):
                load = 0.035 * np.sin(0.25 * t[k] + phases[j])
                th[j, k + 1], om[j, k + 1] = servo_step_np(th[j, k], om[j, k], uk[j], draws[j], h, load)
                om[j, k + 1] += process_noise[j, k]
        err = th[:, 1:] - theta_ref[None, 1:]
        rr = np.sqrt(np.mean(err**2, axis=1))
        mm = np.max(np.abs(err), axis=1)
        uu = np.sqrt(np.mean(vv**2, axis=1))
        metrics[method] = {
            "position_rmse_mean": float(np.mean(rr)),
            "position_rmse_p95": float(np.quantile(rr, 0.95)),
            "max_position_error_mean": float(np.mean(mm)),
            "rms_voltage_mean": float(np.mean(uu)),
            "saturation_fraction_mean": float(np.mean(np.abs(vv) >= 3.199)),
            "rmse_samples": rr.tolist(),
        }
        trajectories[method] = (th[0], vv[0])
    paired = {}
    for other in methods[1:]:
        paired[f"implicit minus {other}"] = paired_bootstrap_ci(metrics["implicit inverse"]["rmse_samples"], metrics[other]["rmse_samples"])

    out = {
        "physical_state_dimension": 2,
        "input_dimension": 1,
        "controlled_output_dimension": 1,
        "forward_velocity_rmse": rms((pred - y0[te]).numpy()),
        "forward_velocity_rmse_control_affine": rms((pred_ca - y0[te]).numpy()),
        "inverse_voltage_rmse_algebraic": rms((ua - u[te]).numpy()),
        "inverse_voltage_rmse_control_affine": rms((uca - u[te]).numpy()),
        "inverse_voltage_rmse_direct": rms((ud - u[te]).numpy()),
        "inverse_contraction_inf": info["kappa_inf"],
        "mean_iterations": float(info["iterations"].float().mean().item()),
        "physics_skip_D_u": float(D_u[0, 0]),
        "state_measurement_noise_std": 0.004,
        "paired_rmse_differences": paired,
        "monte_carlo": metrics,
    }

    fig, ax = plt.subplots(2, 1, figsize=(3.6, 2.5), sharex=True)
    short = {"implicit inverse": "implicit inv.", "control-affine inverse": "control-affine", "direct inverse": "direct inv.", "nominal computed torque": "nominal physics"}
    for method in methods:
        thh, vv = trajectories[method]
        ax[0].plot(t_full[1:], thh[1:], lw=0.8, label=short[method])
        ax[1].plot(t, vv, lw=0.7)
    ax[0].plot(t_full[1:], theta_ref[1:], "k--", lw=0.9, label="reference")
    ax[0].set_ylabel(r"$\varphi$ [rad]", fontsize=7)
    ax[1].set_ylabel("voltage [V]", fontsize=7)
    ax[1].set_xlabel("time [s]", fontsize=7)
    for a in ax:
        a.grid(True, alpha=0.25)
        a.tick_params(labelsize=6)
    handles, labels_ = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.03), columnspacing=0.8, handlelength=1.4)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.955))
    fig.savefig(FIG / "servo_tracking.pdf", bbox_inches="tight")
    fig.savefig(FIG / "servo_tracking.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.boxplot(
        [metrics[m]["rmse_samples"] for m in methods],
        tick_labels=["implicit", "control-affine", "direct", "nominal physics"],
        showfliers=False,
    )
    ax.set_ylabel("position RMSE [rad]")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "servo_monte_carlo.pdf", bbox_inches="tight")
    fig.savefig(FIG / "servo_monte_carlo.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    torch.save({"model": model.state_dict(), "control_affine": model_ca.state_dict(), "inverse": direct.state_dict()}, RES / "servo_models.pt")
    return out


def write_summary(results: dict) -> None:
    rows = ["experiment,metric,value"]
    for exp, vals in results.items():
        for key, value in vals.items():
            if isinstance(value, (int, float)):
                rows.append(f"{exp},{key},{value}")
        for method, metrics in vals.get("monte_carlo", {}).items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    rows.append(f"{exp},{method}:{key},{value}")
        for name, vr in vals.get("variants", {}).items():
            for key, value in vr.items():
                if isinstance(value, (int, float)):
                    rows.append(f"{exp},variant[{name}]:{key},{value}")
            for key, value in vr.get("closed_loop", {}).items():
                if isinstance(value, (int, float)):
                    rows.append(f"{exp},variant[{name}]:closed_loop:{key},{value}")
    (RES / "summary.csv").write_text("\n".join(rows) + "\n")


def main(which: str) -> None:
    results = {}
    p = RES / "all_results.json"
    if p.exists():
        try:
            results = json.loads(p.read_text())
        except Exception:
            results = {}
    if which in ("static", "all"):
        print("[1/3] static map", flush=True)
        results["static"] = run_static()
        print(json.dumps(results["static"], indent=2), flush=True)
        p.write_text(json.dumps(results, indent=2))
    if which in ("narx", "dynamic", "all"):
        print("[2/3] nonlinear NARX plant", flush=True)
        results["narx"] = run_narx()
        print(json.dumps({k: v for k, v in results["narx"].items() if k not in ("monte_carlo", "variants", "H", "S_chi", "S_y", "L_u")}, indent=2), flush=True)
        p.write_text(json.dumps(results, indent=2))
    if which in ("servo", "all"):
        print("[3/3] electromechanical servo", flush=True)
        results["servo"] = run_servo()
        print(json.dumps({k: v for k, v in results["servo"].items() if k != "monte_carlo"}, indent=2), flush=True)
        p.write_text(json.dumps(results, indent=2))
    write_summary(results)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
