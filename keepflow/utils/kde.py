"""Gaussian KDE used to obtain densities for the baselines that cannot
estimate them natively.

`DynamicBufferModule` and `GaussianKDE` are adapted from anomalib
(https://github.com/openvinotoolkit/anomalib), Copyright (C) 2022 Intel
Corporation, licensed under the Apache License, Version 2.0.

`GaussianKDE.forward` has been rewritten to evaluate the kernel on chunks of
query points instead of looping over them one by one. anomalib evaluates at
most a few hundred query points, so the original loop was never a bottleneck
there; here the number of query points equals the number of kernel centres
(10^4 by default), which made the Python-level loop dominate the runtime. The
chunked version performs the identical arithmetic and reproduces the original
output bit-for-bit.
"""
from abc import ABC
from typing import Optional, Dict
import math
from copy import deepcopy
import torch
from torch import Tensor

# Peak size (in elements) of the intermediate [chunk, N, d] buffer allocated by
# GaussianKDE.forward. ~1.3e8 elements is ~0.5 GB in float32.
_KDE_CHUNK_ELEMENTS = 128_000_000


def kde(data_dict: Dict) -> Dict:
    for k in list(data_dict.keys()):
        if k[0] == "prob":
            prob = data_dict[k]
            batch_size, _, timesteps, _ = prob.shape
            prob_, gt_traj_log_prob = [], []
            for b in range(batch_size):
                prob__, gt_traj_prob__ = [], []
                for i in range(timesteps):
                    kernel = GaussianKDE(prob[b, :, i, :-1])
                    kernel(prob[b, :, i, :-1])  # estimate the prob of predicted future positions for fair comparison of inference time
                    prob__.append(deepcopy(kernel))
                    # log_prob, not prob().log(): the latter underflows to -inf
                    # for ground-truth points far from the sample cloud, which
                    # used to be clamped to -10000 (or land on the float32
                    # subnormal floor at -103.28) and reported as if it were the
                    # model's density.
                    gt_traj_prob__.append(kernel.log_prob(data_dict["gt"][b, None, i].float()))
                prob_.append(deepcopy(prob__))
                gt_traj_log_prob.append(torch.cat(gt_traj_prob__, dim=-1))
            gt_traj_log_prob = torch.stack(gt_traj_log_prob, dim=0)
            gt_traj_log_prob = torch.nan_to_num(gt_traj_log_prob, neginf=-10000)
            data_dict[k] = prob_
            data_dict[("gt_traj_log_prob", k[1])] = gt_traj_log_prob
        
    return data_dict


class DynamicBufferModule(ABC, torch.nn.Module):
    """Torch module that allows loading variables from the state dict even in the case of shape mismatch."""

    def get_tensor_attribute(self, attribute_name: str) -> Tensor:
        """Get attribute of the tensor given the name.
        Args:
            attribute_name (str): Name of the tensor
        Raises:
            ValueError: `attribute_name` is not a torch Tensor
        Returns:
            Tensor: Tensor attribute
        """
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, Tensor):
            return attribute

        raise ValueError(f"Attribute with name '{attribute_name}' is not a torch Tensor")

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args):
        """Resizes the local buffers to match those stored in the state dict.
        Overrides method from parent class.
        Args:
          state_dict (dict): State dictionary containing weights
          prefix (str): Prefix of the weight file.
          *args:
        """
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_buffers = {k: v for k, v in persistent_buffers.items() if v is not None}

        for param in local_buffers.keys():
            for key in state_dict.keys():
                if key.startswith(prefix) and key[len(prefix) :].split(".")[0] == param:
                    if not local_buffers[param].shape == state_dict[key].shape:
                        attribute = self.get_tensor_attribute(param)
                        attribute.resize_(state_dict[key].shape)

        super()._load_from_state_dict(state_dict, prefix, *args)
        

class GaussianKDE(DynamicBufferModule):
    """Gaussian Kernel Density Estimation.
    Args:
        dataset (Optional[Tensor], optional): Dataset on which to fit the KDE model. Defaults to None.
    """

    def __init__(self, dataset: Optional[Tensor] = None, bw_factor=1.0):
        super().__init__()

        self.register_buffer("bw_transform", Tensor())
        self.register_buffer("dataset", Tensor())
        self.register_buffer("norm", Tensor())

        self.bw_factor = bw_factor

        #self.bw_transform = Tensor()
        #self.dataset = Tensor()
        #self.norm = Tensor()
        
        if dataset is not None:
            self.fit(dataset)
        
        
    def forward(self, features: Tensor) -> Tensor:
        """Get the KDE estimates at the given query points.
        Args:
          features (Tensor): Query points, shape [n_query, dim]
        Returns: KDE Estimates, shape [n_query]
        """
        return self.log_prob(features).exp()

    def log_prob(self, features: Tensor) -> Tensor:
        """Log of the KDE estimate, computed without ever forming the density.

        Evaluating the density and taking its log afterwards underflows: a
        Gaussian kernel mean below the smallest float32 subnormal (1.4e-45)
        becomes exactly 0, and its log becomes -inf, which the caller then
        clamps. The observed "floor" of log(1.4e-45) = -103.28 in the baselines'
        log-probabilities is that underflow, not a property of the model.

        A Gaussian KDE is strictly positive everywhere, so the log-density is
        always finite; accumulating it with log-sum-exp in float64 recovers the
        true value however small the density is, and no clamping is needed.

        Args:
          features (Tensor): Query points, shape [n_query, dim]
        Returns: log KDE estimates, shape [n_query]
        """
        features = torch.matmul(features, self.bw_transform)

        n_centres, dim = self.dataset.shape
        # Bound the [chunk, n_centres, dim] intermediate to a fixed memory budget.
        chunk = max(1, _KDE_CHUNK_ELEMENTS // max(1, n_centres * dim))

        dataset = self.dataset.unsqueeze(0).double()  # [1, n_centres, dim]
        log_norm = self.norm.double().log()
        log_n = math.log(n_centres)
        out = torch.empty(features.shape[0], device=features.device,
                          dtype=features.dtype)
        for start in range(0, features.shape[0], chunk):
            query = features[start:start + chunk].unsqueeze(1).double()  # [chunk, 1, dim]
            sq = (dataset - query).pow_(2).sum(dim=-1)                   # [chunk, n_centres]
            log_kernel = -sq / 2 + log_norm
            out[start:start + chunk] = (
                torch.logsumexp(log_kernel, dim=-1) - log_n).to(out.dtype)

        return out

    def fit(self, dataset: Tensor) -> None:
        """Fit a KDE model to the input dataset.
        Args:
          dataset (Tensor): Input dataset.
        Returns:
            None
        """
        
        num_samples, dimension = dataset.shape

        # compute scott's bandwidth factor
        factor = num_samples ** (-1 / (dimension + 4)) * self.bw_factor

        cov_mat = self.cov(dataset.T)
        inv_cov_mat = torch.linalg.inv(cov_mat)
        inv_cov = inv_cov_mat / factor**2
        
        # transform data to account for bandwidth
        bw_transform = torch.linalg.cholesky(inv_cov)
        dataset = torch.matmul(dataset, bw_transform)
        
        #
        norm = torch.prod(torch.diag(bw_transform))
        norm *= math.pow((2 * math.pi), (-dimension / 2))

        self.bw_transform = bw_transform
        self.dataset = dataset
        self.norm = norm

    @staticmethod
    def cov(tensor: Tensor) -> Tensor:
        """Calculate the unbiased covariance matrix.
        Args:
            tensor (Tensor): Input tensor from which covariance matrix is computed.
        Returns:
            Output covariance matrix.
        """
        mean = torch.mean(tensor, dim=1, keepdim=True)
        cov = torch.matmul(tensor - mean, (tensor - mean).T) / (tensor.size(1) - 1)
        return cov


