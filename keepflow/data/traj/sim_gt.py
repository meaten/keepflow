"""Exact ground-truth posterior for the SimTrajDensity fork and branch subsets.

The generator (`process_data.py`) builds K noise-free trajectories (2 for fork,
4 for branch) that share the same straight history ending at the origin, picks
one uniformly per sequence, and adds i.i.d. N(0, s^2) noise (s = 0.3) to all
L + T positions, observed ones included. Evaluation works in the frame centred
on the *noisy* last observation, so with e_* the per-position noise,

    x_n = mu_{n,k} + e_n - e_L,     n = 1..T.

The history fixes d_j = e_j - e_L (deviation of observation j from the
noise-free straight line), and with m = L - 1 such deviations
e_L | d ~ N(-sum(d)/L, s^2/L). Per coordinate, therefore,

    x | o, k ~ N(mu_k + sum(d)/L, s^2 I_T + (s^2/L) 11^T),

and the mode is not identified by the shared history. Observing a prefix of
the future (the update rows) is ordinary Gaussian conditioning within each
mode, with the mode weights re-weighted by the prefix likelihood.

The GMM stored in the test pickle (`env.gt_dist`) has the right mode means but
a per-step std of 0.09 (= s^2: `scale`, already equal to s, is multiplied by s
again) and ignores the shared offset; only its means and weights are used here.
"""
import numpy as np
import torch

FORK_NOISE_STD = 0.3  # process_data.py, simfork/simbranch generator


class ExactForkBranchGT:
    def __init__(self, stored_gmm, obs_len: int = 8, sigma: float = FORK_NOISE_STD):
        comp = stored_gmm.component_distribution.base_dist
        self.mu = comp.loc.double().cpu()                            # (T, K, 2)
        logits = stored_gmm.mixture_distribution.logits.double().cpu()
        assert torch.allclose(logits, logits[:1].expand_as(logits)), \
            "expected one mode per trajectory, shared across steps"
        self.log_w0 = torch.log_softmax(logits[0], -1)               # (K,)
        self.sigma = sigma
        self.L = obs_len
        T = self.mu.shape[0]
        s2 = sigma ** 2
        self.cov = s2 * torch.eye(T, dtype=torch.float64) + \
            s2 / obs_len * torch.ones(T, T, dtype=torch.float64)     # per coordinate
        self.shift = None

    # ------------------------------------------------------------------ #
    def condition(self, obs_xy, prefix=None):
        """obs_xy: (L, 2) observed positions in the evaluation frame (last one
        at the origin). prefix: optional (k, 2) observed future positions.
        Sets the posterior over the remaining T - k steps."""
        obs_xy = torch.as_tensor(obs_xy, dtype=torch.float64).cpu()
        L = obs_xy.shape[0]
        assert L == self.L, (L, self.L)
        line = torch.stack([torch.arange(-(L - 1), 1, dtype=torch.float64),
                            torch.zeros(L, dtype=torch.float64)], dim=1)
        d = (obs_xy - line)[:-1]
        self.shift = d.sum(0) / L                                    # E[-e_L | d]
        mean = self.mu + self.shift                                  # (T, K, 2)
        T = mean.shape[0]
        k = 0 if prefix is None else int(prefix.shape[0])
        self.k = k
        if k == 0:
            self.log_w = self.log_w0.clone()
            self.mean_r = mean                                       # (T, K, 2)
            self.cov_r = self.cov
        else:
            prefix = torch.as_tensor(prefix, dtype=torch.float64).cpu()
            So, Sr = self.cov[:k, :k], self.cov[k:, k:]
            Sro = self.cov[k:, :k]
            A = Sro @ torch.linalg.inv(So)                           # (T-k, k)
            resid = prefix[:, None, :] - mean[:k]                    # (k, K, 2)
            self.mean_r = mean[k:] + torch.einsum("rk,kmd->rmd", A, resid)
            self.cov_r = Sr - A @ Sro.T
            mvn = torch.distributions.MultivariateNormal(
                torch.zeros(k, dtype=torch.float64), covariance_matrix=So)
            ll = mvn.log_prob(resid.permute(1, 2, 0)).sum(-1)        # (K,)
            self.log_w = torch.log_softmax(self.log_w0 + ll, -1)
        self.chol_r = torch.linalg.cholesky(self.cov_r)
        self.step_sd = torch.sqrt(torch.diagonal(self.cov_r))        # (T-k,)
        return self

    # ------------------------------------------------------------------ #
    def step_logp(self, pts):
        """pts: (B, T-k, 2) -> (B, T-k) per-step marginal log density."""
        pts = torch.as_tensor(pts, dtype=torch.float64).cpu()
        logn = torch.distributions.Normal(self.mean_r[None], self.step_sd[None, :, None, None]) \
            .log_prob(pts[:, :, None]).sum(-1)                       # (B, T-k, K)
        return torch.logsumexp(logn + self.log_w, dim=-1)

    def grid_density(self, query):
        """query: (G, 2) points -> (G, T-k) per-step marginal density."""
        query = torch.as_tensor(query, dtype=torch.float64).cpu()
        Tr = self.mean_r.shape[0]
        pts = query[:, None].expand(-1, Tr, -1)
        return torch.exp(self.step_logp(pts))

    def joint_logp(self, seq):
        """seq: (B, T-k, 2) -> (B,) joint log density of the remaining steps."""
        seq = torch.as_tensor(seq, dtype=torch.float64).cpu()
        mvn = torch.distributions.MultivariateNormal(
            torch.zeros(self.mean_r.shape[0], dtype=torch.float64), scale_tril=self.chol_r)
        r = seq[:, None] - self.mean_r.transpose(0, 1)[None]          # (B, K, T-k, 2)
        logn = mvn.log_prob(r.transpose(-1, -2)).sum(-1)             # (B, K)
        return torch.logsumexp(logn + self.log_w, dim=-1)

    def sample(self, n, rng):
        """(n, T-k, 2) exact draws from the conditioned posterior."""
        K = self.mean_r.shape[1]
        modes = rng.choice(K, size=n, p=torch.exp(self.log_w).numpy())
        eps = torch.as_tensor(rng.standard_normal((n, self.mean_r.shape[0], 2)))
        noise = torch.einsum("ts,nsd->ntd", self.chol_r, eps)
        return self.mean_r[:, modes].transpose(0, 1) + noise
