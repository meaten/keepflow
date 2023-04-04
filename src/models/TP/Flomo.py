from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from utils import optimizer_to_device


class RNN(nn.Module):
    """ GRU based recurrent neural network. """

    def __init__(self, nin, nout, es=16, hs=16, nl=3, device=0):
        super().__init__()
        self.embedding = nn.Linear(nin, es)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.output_layer = nn.Linear(hs, nout)
        self.device = device
        self.cuda(self.device)

    def forward(self, x, hidden=None):
        x = F.relu(self.embedding(x))
        x, hidden = self.gru(x, hidden)
        x = self.output_layer(x)
        return x, hidden

class FloMo(nn.Module):

    def __init__(self, cfg, device=0):
        super().__init__()
        self.device = cfg.DEVICE
        
        self.output_path = Path(cfg.OUTPUT_DIR)
        
        self.pred_steps = cfg.DATA.PREDICT_LENGTH
        self.output_size = self.pred_steps * 2
        if cfg.DATA.DATASET_NAME == "sdd":
            self.alpha = 10
            self.beta = 0.2
            self.gamma = 0.02
        else:
            self.alpha = 3
            self.beta = 0.002
            self.gamma = 0.002
        self.B = 15
        self.rel_coords = True
        self.norm_rotation = False

        # core modules
        self.obs_encoding_size = 16
        self.obs_encoder = RNN(nin=2, nout=self.obs_encoding_size, device=device)
        self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.obs_encoding_size, n_layers=10, K=8,
                                    B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)

        # move model to specified device
        self.device = device
        self.to(device)
        
        self.optimizer = torch.optim.Adam(
            self.parameters(), 1e-3)
        self.optimizers = [self.optimizer]

    def _encode_conditionals(self, x):
        # encode observed trajectory
        x_in = x
        if self.rel_coords:
            x_in = x[:,1:] - x[:,:-1] # convert to relative coords
        x_enc, hidden = self.obs_encoder(x_in) # encode relative histories
        x_enc = x_enc[:,-1]
        x_enc_context = x_enc
        return x_enc_context

    def _abs_to_rel(self, y, x_t):
        y_rel = y - x_t # future trajectory relative to x_t
        y_rel[:,1:] = (y_rel[:,1:] - y_rel[:,:-1]) # steps relative to each other
        y_rel = y_rel * self.alpha # scale up for numeric reasons
        return y_rel

    def _rel_to_abs(self, y_rel, x_t):
        y_abs = y_rel / self.alpha
        return torch.cumsum(y_abs, dim=-2) + x_t 

    def _rotate(self, x, x_t, angles_rad):
        c, s = torch.cos(angles_rad), torch.sin(angles_rad)
        c, s = c.unsqueeze(1), s.unsqueeze(1)
        x_center = x - x_t # translate
        x_vals, y_vals = x_center[:,:,0], x_center[:,:,1]
        new_x_vals = c * x_vals + (-1 * s) * y_vals # _rotate x
        new_y_vals = s * x_vals + c * y_vals # _rotate y
        x_center[:,:,0] = new_x_vals
        x_center[:,:,1] = new_y_vals
        return x_center + x_t # translate back

    def _normalize_rotation(self, x, y_true=None):
        x_t = x[:,-1:,:]
        # compute rotation angle, such that last timestep aligned with (1,0)
        x_t_rel = x[:,-1] - x[:,-2]
        rot_angles_rad = -1 * torch.atan2(x_t_rel[:,1], x_t_rel[:,0])
        x = self._rotate(x, x_t, rot_angles_rad)
        
        if y_true != None:
            y_true = self._rotate(y_true, x_t, rot_angles_rad)
            return x, y_true, rot_angles_rad # inverse

        return x, rot_angles_rad # forward pass

    def _inverse(self, y_true, x):
        if self.norm_rotation:
            x, y_true, angle = self._normalize_rotation(x, y_true)

        x_t = x[...,-1:,:]
        x_enc = self._encode_conditionals(x) # history encoding
        y_rel = self._abs_to_rel(y_true, x_t)
        y_rel_flat = torch.flatten(y_rel, start_dim=1)

        if self.training:
            # add noise to zero values to avoid infinite density points
            zero_mask = torch.abs(y_rel_flat) < 1e-2 # approx. zero
            noise = torch.randn_like(y_rel_flat) * self.beta
            y_rel_flat = y_rel_flat + (zero_mask * noise)

            # minimally perturb remaining motion to avoid x1 = x2 for any values
            noise = torch.randn_like(y_rel_flat) * self.gamma
            y_rel_flat = y_rel_flat + (~zero_mask * noise)
        
        z, jacobian_det = self.flow.inverse(torch.flatten(y_rel_flat, start_dim=1), x_enc)
        return z, jacobian_det

    def _repeat_rowwise(self, c_enc, n):
        org_dim = c_enc.size(-1)
        c_enc = c_enc.repeat(1, n)
        return c_enc.view(-1, n, org_dim)

    def forward(self, z, c):
        raise NotImplementedError

    def sample(self, n, x):
        with torch.no_grad():
            if self.norm_rotation:
                x, rot_angles_rad = self._normalize_rotation(x)
            x_enc = self._encode_conditionals(x) # history encoding
            x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size)
            n_total = n * x.size(0)
            output_shape = (x.size(0), n, self.pred_steps, 2) # predict n trajectories input

            # sample and compute likelihoods
            z = torch.randn([n_total, self.output_size]).to(self.device)
            samples_rel, log_det = self.flow(z, x_enc_expanded)
            samples_rel = samples_rel.view(*output_shape)
            normal = Normal(0, 1, validate_args=True)
            log_probs = (normal.log_prob(z).sum(1) - log_det).view((x.size(0), -1))

            x_t = x[...,-1:,:].unsqueeze(dim=1).repeat(1, n, 1, 1)
            samples_abs = self._rel_to_abs(samples_rel, x_t)

            # invert rotation normalization
            if self.norm_rotation:
                x_t_all = x[...,-1,:]
                for i in range(len(samples_abs)):
                    pred_trajs = samples_abs[i]
                    samples_abs[i] = self._rotate(pred_trajs, x_t_all[i], -1 * rot_angles_rad[i].repeat(pred_trajs.size(0)))

            return samples_abs, log_probs

    def log_prob(self, y_true, x):
        z, log_abs_jacobian_det = self._inverse(y_true, x)
        normal = Normal(0, 1, validate_args=True)
        return normal.log_prob(z).sum(1) + log_abs_jacobian_det
    
    def predict(self, data_dict, return_prob=False):
        x = data_dict["obs_st"]
        n = 10000 if return_prob else 1
        traj_pred, _ = self.sample(n, x)
        data_dict[("pred_st", 0)] = traj_pred[:, 0]
        
        if return_prob:
            traj_pred = torch.cat([traj_pred, torch.zeros_like(traj_pred)], dim=3)
            data_dict[("prob_st", 0)] = traj_pred[..., :3]
            
        return data_dict
    
    def predict_from_new_obs(self, data_dict, time_step: int):
        return data_dict
    
    def update(self, data_dict):
        x = data_dict["obs_st"]
        y_true = data_dict["gt_st"]
        train_loss = -self.log_prob(y_true, x).mean()
        
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        
        return {"loss": train_loss.item()}
    
    def save(self, epoch: int = 0, path: Path = None) -> None:
        if path is None:
            path = self.output_path / "ckpt.pt"

        ckpt = {
            'epoch': epoch,
            'state': self.state_dict(),
            'optim_state': self.optimizer.state_dict(),
        }

        torch.save(ckpt, path)

    def check_saved_path(self, path: Path = None) -> bool:
        if path is None:
            path = self.output_path / "ckpt.pt"

        return path.exists()

    def load(self, path: Path = None) -> int:
        if path is None:
            path = self.output_path / "ckpt.pt"

        ckpt = torch.load(path)
        self.load_state_dict(ckpt['state'])

        self.optimizer.load_state_dict(ckpt['optim_state'])
        optimizer_to_device(self.optimizer, self.device)

        return ckpt["epoch"]
    
    
class FCN(nn.Module):
    """ Simple fully connected network. """

    def __init__(self, nin, nout, nh=[24, 24, 24], device=0):
        super().__init__()
        if type(nh) != list:
            nh = [nh] * 3
        self.layers = [nin] + nh + [nout]
        self.net = []
        for (l1, l2) in zip(self.layers, self.layers[1:]):
            self.net.extend([nn.Linear(l1, l2), nn.ELU()])
        self.net.pop() # remove last activation
        self.net = nn.Sequential(*self.net)
        self.device = device
        self.cuda(self.device)

    def forward(self, x):
        return self.net(x)

class FlowSequential(nn.Sequential):
    """ Container for normalizing flow layers. """
    
    def forward(self, x, c=None):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, c)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, c=None):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, c)
            sum_log_abs_det_jacobians += log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians

class InvertiblePermutation(nn.Module):
    """
    Randomly permutes inputs and implments the reverse operation.
    Returns 0 for log absolute jacobian determinant, as the jacobian
    determinant of a permutation is 1 (orthogonal matrix) and log(1)
    is 0.
    """

    def __init__(self, dim, reverse_only=False):
        super().__init__()
        self.dim = dim
        self.reverse_only = reverse_only
        self.register_buffer('perm_idx', torch.randperm(dim))
        self.register_buffer('perm_idx_inv', torch.zeros_like(self.perm_idx))
        
        # initialize perm_idx_inv to reverse perm_idx sorting
        for i, j in zip(self.perm_idx, torch.arange(self.dim)):
            self.perm_idx_inv[i] = j

    def forward(self, x, c=None):
        if self.reverse_only:
            return x.flip(-1), 0
        x = x[..., self.perm_idx]
        return x, 0

    def inverse(self, x, c=None):
        if self.reverse_only:
            return x.flip(-1), 0
        x = x[..., self.perm_idx_inv]
        return x, 0

class CouplingNSF(nn.Module):
    """
    Neural spline flow with coupling conditioner [Durkan et al. 2019].
    """
    def __init__(self, dim, dimc=0, K=5, B=3, hidden_dim=8, device=0):
        super().__init__()
        self.dim = dim
        self.dimc = dimc # conditionals dim
        self.K = K # number of knots
        self.B = B # spline support
        # output: for each input dim params that define one spline
        self.conditioner = FCN(dim // 2 + self.dimc, (3 * K - 1) * (self.dim - (self.dim // 2)), hidden_dim)
        self.device = device
        self.to(self.device)

    def _get_spline_params(self, x1, c=None):
        x = x1 if c == None else torch.cat((x1, c), -1) # concat inputs
        out = self.conditioner(x).view(-1, self.dim - (self.dim // 2), 3 * self.K - 1) # calls f(x_1:d), arange spline params by input dim
        W, H, D = torch.split(out, self.K, dim = 2) # get knot width, height, derivatives
        return W, H, D

    def forward(self, x, c=None):
        # compute spline parameters
        x1, x2 = x[:, :self.dim // 2], x[:, self.dim // 2:] # split input
        W, H, D = self._get_spline_params(x1, c)
        # apply spline transform
        x2, ld = unconstrained_RQS(x2, W, H, D, inverse=False, tail_bound=self.B)
        log_det = torch.sum(ld, dim = 1)
        return torch.cat([x1, x2], dim = 1), log_det

    def inverse(self, z, c=None):
        # compute spline parameters
        z1, z2 = z[:, :self.dim // 2], z[:, self.dim // 2:]
        W, H, D = self._get_spline_params(z1, c)
        # apply spline transform
        z2, ld = unconstrained_RQS(z2, W, H, D, inverse=True, tail_bound=self.B)
        log_det = torch.sum(ld, dim = 1)
        return torch.cat([z1, z2], dim = 1), log_det

class NeuralSplineFlow(nn.Module):

    def __init__(self, nin, nc=0, n_layers=5, K=5, B=3, hidden_dim=8, device=0):
        super().__init__()
        self.nin = nin
        self.nc = nc # size of conditionals
        self.n_layers = n_layers
        
        self.net = []
        for i in range(self.n_layers):
            self.net.append(CouplingNSF(self.nin, self.nc, K, B, hidden_dim=hidden_dim, device=device))
            self.net.append(InvertiblePermutation(self.nin, reverse_only=False))
        self.net.pop()
        self.net = FlowSequential(*self.net)

        self.device = device
        self.to(self.device)

    def forward(self, z, c=None):
        return self.net(z, c)

    def inverse(self, x, c=None):
        return self.net.inverse(x, c)

    def sample(self, n, c=None):
        with torch.no_grad():
            if c != None:
                assert(c.size(0) == n)
            samples = torch.zeros([n, self.nin]).to(self.device)
            z = torch.randn_like(samples)
            x, log_det = self.forward(z, c)
            return x
    
    def log_prob(self, x, c=None):
        """
        Computes the log likelihood of the input. The likelihood of z can be
        evaluted as a log sum, because the product of univariate normals gives
        a multivariate normal with diagonal covariance.
        """
        z, log_abs_jacobian_det = self.inverse(x, c)
        normal = Normal(0, 1, validate_args=True)
        return normal.log_prob(z).sum(1) + log_abs_jacobian_det
    
    
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6): # bisection search
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1

# Unconstrained Rational-Quadratic Spline
def unconstrained_RQS(inputs, unnormalized_widths, unnormalized_heights,
                      unnormalized_derivatives, inverse=False,
                      tail_bound=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound) # mask for values within support
    outside_interval_mask = ~inside_intvl_mask

    outputs = torch.zeros_like(inputs) # prepare outputs
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask] # apply linear function trails
    logabsdet[outside_interval_mask] = 0

    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
        inputs=inputs[inside_intvl_mask],
        unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )
    return outputs, logabsdet

def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse=False, left=0., right=1.,
        bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE):
    if torch.min(inputs) < left or torch.max(inputs) > right: # assert all values within support
        raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths # make every width at least min_bin_width, but keep total sum of 1
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0) # zero padding left
    cumwidths = (right - left) * cumwidths + left # scale on support and shift s.t. it covers [-B, B]
    cumwidths[..., 0] = left # fix left and right value to the boundaries
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1] # compute actual widths from the cummulative form

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse: # bisection search for the correct bin
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0] # get matched cumulative widths and widths
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0] # get matched cumulative heights and heights
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    delta = heights / widths # equals s^k in formula
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0] # get matched derivatives
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx) # index moved +1 basically to get derivative of knot k+1
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    if inverse: # compute the inverse spline transformation
        a = (((inputs - input_cumheights) * (input_derivatives \
            + input_derivatives_plus_one - 2 * input_delta) \
            + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights) \
            * (input_derivatives + input_derivatives_plus_one \
            - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta \
                      + ((input_derivatives + input_derivatives_plus_one \
                      - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * root.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet # inverse function theorem and log(x^-1) = -1 * log(x)
    else: # compute the forward spline transformation
        theta = (inputs - input_cumwidths) / input_bin_widths # equals xi in formula
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) \
                    + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives \
                      + input_derivatives_plus_one - 2 * input_delta) \
                      * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * theta.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet