"""Exact ground-truth density for the SimTrajDensity *unicycle* subset.

The unicycle generator (`process_data.py`) draws exactly two scalars per
trajectory,

    a ~ U(-1, 1)   perturbs the first step's speed:  v_0 = 1 + c * a
    b ~ U(-1, 1)   sets the fan angle:               angle_m = b * theta_m

with `theta_m = (m + 1) * max_ang_acc`, and forms

    vel_m = v_m * [cos(b * theta_m), sin(b * theta_m)],   v_m = 1 for m >= 1
    pos_n = sum_{m <= n} vel_m .

So the position at step n is the push-forward of a 2-D uniform through a
smooth map F_n: (a, b) -> (x, y), and the density follows exactly from the
change-of-variables formula

    p_n(x, y) = (1/4) / |det J_{F_n}(a, b)|,    (a, b) = F_n^{-1}(x, y),

and 0 outside the image of the square. The Jacobian determinant is available
in closed form,

    det J_n(a, b) = c * [ v_0 * theta_0
                          + sum_{m=1..n} theta_m * cos(b * (theta_m - theta_0)) ],

which is strictly positive over the domain (the angle differences stay below
~0.96 rad, so every cosine exceeds 0.57), hence F_n is injective and the
preimage is unique.

This replaces the previous kernel-density approximation of the ground truth.
KDE is a poor fit here: the true density has a hard support boundary at the
edges of the fan and a 1/|det J| profile inside it, which a Gaussian kernel
necessarily smooths out. Since the paper's own argument is that post-hoc KDE
is an inaccurate way to obtain densities, using KDE for the *evaluation
ground truth* undercut the benchmark.
"""
import numpy as np
import torch

# Must match process_data.py's unicycle generator.
PRED_LEN = 12
MAX_ANG_ACC = np.pi / 3 / PRED_LEN
MAX_POS_CHANGE = 0.1
THETA = (np.arange(PRED_LEN) + 1) * MAX_ANG_ACC


class AnalyticUnicycleDensity:
    """Exact marginal density of the unicycle position at one prediction step.

    Call with a (N, 2) tensor of query points; returns a (N,) tensor of
    densities, matching the call signature of `keepflow.utils.kde.GaussianKDE`
    so it can be used wherever the KDE ground truth was used.
    """

    def __init__(self, step: int, n_newton: int = 60):
        self.step = int(step)
        self.n_newton = n_newton
        self.theta = THETA[: self.step + 1]

    # --- forward map and its derivatives -------------------------------
    def _spine(self, b):
        """Position for a = 0, i.e. all speeds equal to 1. b: (N,)"""
        ang = b[:, None] * self.theta[None, :]
        return np.stack([np.cos(ang).sum(1), np.sin(ang).sum(1)], axis=1)

    def _dir(self, b):
        """Direction along which `a` displaces the point (scaled by c)."""
        return np.stack([np.cos(b * self.theta[0]),
                         np.sin(b * self.theta[0])], axis=1) * MAX_POS_CHANGE

    def _residual(self, b, xy):
        """Cross product of (P - spine(b)) with dir(b); zero at the solution."""
        S = self._spine(b)
        d = self._dir(b)
        return (xy[:, 0] - S[:, 0]) * d[:, 1] - (xy[:, 1] - S[:, 1]) * d[:, 0]

    def det_jacobian(self, a, b):
        v0 = 1.0 + MAX_POS_CHANGE * a
        term = v0 * self.theta[0]
        if self.step >= 1:
            term = term + (self.theta[1:] *
                           np.cos(b[:, None] * (self.theta[1:] - self.theta[0])[None, :])
                           ).sum(1)
        return MAX_POS_CHANGE * term

    # --- inverse map ----------------------------------------------------
    def invert(self, xy):
        """Bisection on b (the residual is monotone over the domain), then a
        directly by projection. Returns (a, b, valid)."""
        lo = np.full(xy.shape[0], -1.0)
        hi = np.full(xy.shape[0], 1.0)
        f_lo = self._residual(lo, xy)
        f_hi = self._residual(hi, xy)
        # no sign change -> the point lies outside the fan
        valid = (f_lo * f_hi) <= 0
        for _ in range(self.n_newton):
            mid = 0.5 * (lo + hi)
            f_mid = self._residual(mid, xy)
            go_left = (f_lo * f_mid) <= 0
            hi = np.where(go_left, mid, hi)
            lo = np.where(go_left, lo, mid)
            f_lo = np.where(go_left, f_lo, f_mid)
        b = 0.5 * (lo + hi)
        S = self._spine(b)
        d = self._dir(b)
        a = ((xy - S) * d).sum(1) / (d * d).sum(1)
        valid = valid & (a >= -1.0) & (a <= 1.0)
        return a, b, valid

    def __call__(self, query):
        as_torch = torch.is_tensor(query)
        device = query.device if as_torch else None
        xy = query.detach().cpu().numpy() if as_torch else np.asarray(query)
        xy = xy[:, :2].astype(np.float64)

        a, b, valid = self.invert(xy)
        dens = np.zeros(xy.shape[0], dtype=np.float64)
        if valid.any():
            det = np.abs(self.det_jacobian(a[valid], b[valid]))
            dens[valid] = 0.25 / np.maximum(det, 1e-300)

        if as_torch:
            return torch.as_tensor(dens, dtype=query.dtype, device=device)
        return dens


def build_unicycle_gt(pred_len: int = PRED_LEN):
    """One exact density per prediction step."""
    return [AnalyticUnicycleDensity(step) for step in range(pred_len)]


def sample_unicycle(n_samples: int, pred_len: int = PRED_LEN, seed: int = 0):
    """Draw exact samples of the future positions, shape (n_samples, pred_len, 2).

    Used for the conditional (post-update) ground truth, where the evaluation
    conditions on the observed prefix by keeping the trajectories closest to
    it; drawing many samples from the exact generator is strictly better than
    reusing the 1000 trajectories that happen to be in the test split.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, n_samples)
    b = rng.uniform(-1.0, 1.0, n_samples)
    ang = b[:, None] * THETA[None, :pred_len]
    vx = np.cos(ang)
    vy = np.sin(ang)
    vx[:, 0] *= (1.0 + MAX_POS_CHANGE * a)
    vy[:, 0] *= (1.0 + MAX_POS_CHANGE * a)
    return np.stack([np.cumsum(vx, axis=1), np.cumsum(vy, axis=1)], axis=-1)
