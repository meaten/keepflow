import sys
import logging

import torch
import torch.optim as optim

from einops import rearrange

from keepflow.models import ModelTemplate
sys.path.append('extern/traj/MoFlow')
from eval_eth import build_network as build_network_fm
from imle_eth import build_network as build_network_imle
from utils.config import Config
from utils.normalization import unnormalize_min_max, normalize_min_max


class MoFlow(ModelTemplate):
    def __init__(self, config):
        """
        changes: hyperparameters
        """

        super().__init__(config)

        self.config = config

        cfg, logger, _ = init_basics_fm(config.DATA.DATASET_NAME)
        
        self.cfg = cfg

        self.denoiser = build_network_fm(cfg, None, logger)

        from ema_pytorch import EMA
        self.ema = EMA(self.denoiser, beta=0.995, update_every=1)

        self.optimizer = optim.AdamW([{'params': self.denoiser.parameters()}
                                     ],
                                    lr=self.cfg.OPTIMIZATION.LR,
                                    weight_decay=0.01)
        
        self.optimizers = [self.optimizer]

    def create_batch(self, data_dict):
        batch = {}
        # batch['indexes'] = torch.stack([i[1] for i in data_dict['index']])[:, None]
        batch['batch_size'] = data_dict['obs'].shape[0]
        
        # only original scale is used for past trajectory
        batch['past_traj_original_scale'] = torch.cat(
            [data_dict['obs'][..., :2] + data_dict['standardize_offset'][..., :2], # not centered
             data_dict['obs'][..., :4]],  # centered pos, vel
             dim=-1
             )[:, None]
        
        # use scaled future trajectory
        batch['fut_traj'] = normalize_min_max(data_dict['gt'][:, None], self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1)
        
        return  batch
    
    def forward(self, data_dict):
        pass

    def training_step(self, data_dict):
        batch = self.create_batch(data_dict)
        
        loss, loss_reg, loss_cls, loss_vel = self.denoiser(batch, {'cur_epoch': 0})
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.ema.update()
        
        return {"loss": loss.item()}
   
    def predict(self, data_dict, return_prob=False):
        
        batch = self.create_batch(data_dict)
        
        # cannot change num_samples. stochastic sampling
        n_sample = 20
        
        # pred_traj, pred_traj_at_t, t_seq, y_t_seq, pred_score = self.denoiser.sample(batch, num_trajs=n_sample, return_all_states=False)
        pred_traj, pred_traj_at_t, t_seq, y_t_seq, pred_score = self.ema.ema_model.sample(batch, num_trajs=n_sample, return_all_states=False)
        
        assert list(pred_traj.shape[2:]) == [self.cfg.agents, self.cfg.MODEL.MODEL_OUT_DIM]

        pred_traj = rearrange(pred_traj, 'b k a (f d) -> (b a) k f d', f=self.pred_len)[...,0:2]  # [B, k_preds, 11, 40] -> [B * 11, k_preds, 20, 2]
        # pred_traj_at_t = rearrange(pred_traj_at_t, 'b t k a (f d) -> (b a) t k f d', f=self.pred_len)[...,0:2]  # [B, k_preds, 11, 40] -> [B * 11, k_preds, 20, 2]

        pred_traj = unnormalize_min_max(pred_traj, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1) 
        
        pred_traj = pred_traj.clone()

        data_dict[("pred", 0)] = pred_traj
        
        if return_prob:
            trajs = [data_dict[("pred", 0)]]
            # n_sample trajectories per call; repeat until the sample budget is met.
            n_repeat = max(0, -(-self.kde_n_sample // n_sample) - 1)
            for _ in range(n_repeat):
                pred_traj, pred_traj_at_t, t_seq, y_t_seq, pred_score = self.ema.ema_model.sample(batch, num_trajs=n_sample, return_all_states=False)
                assert list(pred_traj.shape[2:]) == [self.cfg.agents, self.cfg.MODEL.MODEL_OUT_DIM]
                pred_traj = rearrange(pred_traj, 'b k a (f d) -> (b a) k f d', f=self.pred_len)[...,0:2]  # [B, k_preds, 11, 40] -> [B * 11, k_preds, 20, 2]
                pred_traj = unnormalize_min_max(pred_traj, self.cfg.fut_traj_min, self.cfg.fut_traj_max, -1, 1) 
                pred_traj = pred_traj.clone()
                trajs.append(pred_traj)
                
            trajs = torch.cat(trajs, dim=1)
            trajs = torch.cat([trajs, torch.zeros_like(trajs)], dim=3)
            data_dict[("prob", 0)] = trajs[..., :3]
        
        return data_dict

class MoFlow_IMLE(ModelTemplate):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        
        cfg, logger, _ = init_basics_imle(config.DATA.DATASET_NAME)
        args = DummyArgs(cfg)
        self.denoiser = build_network_imle(cfg, args, logger)

    def forward(self, data_dict):
        pass

    def training_step(self, data_dict):
        pass
        
    def predict(self, data_dict, return_prob=False):
        pass



def init_basics_fm(subset):
    """
    Init the basic configurations for the experiment.
    """

    """Load the config file"""
    if subset in ["eth", "univ", "hotel", "zara1", "zara2"]:
        dir_name = 'eth_ucy'
    elif subset == "sdd":
        dir_name = 'sdd'
    else:
        dir_name = 'eth_ucy'
        
    cfg = Config(f'extern/traj/MoFlow/cfg/{dir_name}/cor_fm.yml', 'reproduce')

    dict_min_max = {
        "eth": [-10.0525, 7.4460],
        "univ": [-11.5357, 7.4460],
        "hotel": [-11.5357, 6.9198],
        "zara1": [-11.5357, 7.4460],
        "zara2": [-10.0525, 7.4460],
        "simfork": [-6.0, 12.0],
        "simbranch": [-6.0, 12.0],
        "simunicycle": [-6.0, 12.0],
    }

    cfg.fut_traj_min = dict_min_max[subset][0]
    cfg.fut_traj_max = dict_min_max[subset][1]

    tag = '_'

    if cfg.denoising_method == 'fm':
        cfg.sigma_data = 0.13
        cfg.sampling_steps = 10
        
        cfg.t_schedule = 'logit_normal'

        cfg.logit_norm_mean = -0.5
        cfg.logit_norm_std = 1.5

        cfg.fm_wrapper = 'direct'
        cfg.fm_rew_sqrt = False
        cfg.fm_in_scaling = True

        cfg.drop_method = 'emb'
        cfg.drop_logi_k = 20.0
        cfg.drop_logi_m = 0.5
        tag += f'_drop_{cfg.drop_method}_m{cfg.drop_logi_m}_k{cfg.drop_logi_k}'

        tag += '_IS'

        cfg.solver = 'lin_poly'
        cfg.lin_poly_p = 5
        cfg.lin_poly_long_step = 1000
        solver_tag_ = f'lin_poly_p{cfg.lin_poly_p}_long{cfg.lin_poly_long_step}'
        
        fm_tag_ = f'FM_S{cfg.sampling_steps}_{solver_tag_}'
        tag += fm_tag_
        cfg.solver_tag = fm_tag_

    cfg.MODEL.USE_PRE_NORM = False
    cfg.MODEL.NUM_LAYERS = None
    cfg.MODEL.DROPOUT = None
    
    cfg.tied_noise = True
    if cfg.tied_noise:
        tag += '_TN'

    cfg.LOSS_NN_MODE = 'agent'
    cfg.LOSS_REG_REDUCTION = 'sum'
    cfg.LOSS_REG_SQUARED = False
    cfg.LOSS_VELOCITY = False

    tag += f'_NN_{cfg.LOSS_NN_MODE[:1].upper()}'
    tag += f'_REG_{cfg.LOSS_REG_REDUCTION[:1].upper()}'

    cfg.subset = subset
    tag += f'_{cfg.subset}'

    cfg.rotate = True
    cfg.rotate_time_frame = 6
    cfg.rotate_aug = False
    tag += f'_rot_{cfg.rotate_time_frame}'

    # change: 'min_max' -> 'original'
    cfg.data_norm = 'original'
    tag += f'_{cfg.data_norm}'

    # override the batch size
    cfg.train_batch_size = 32
    cfg.test_batch_size = 32

    cfg.OPTIMIZATION.LR = 1e-4
    cfg.MODEL.CONTEXT_ENCODER.AGENTS = 1
    cfg.agents = 1

    ### voila, create the saving directory ###
    tag += '_train_set' if True else '_test_set'
    tag = tag.replace('__', '_')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = logging.getLogger(__name__)

    for key in sorted(dir(cfg)):
        if not key.startswith('__') and not callable(getattr(cfg, key)):
            if key in ['yml_dict', 'logger']:
                continue
            if key not in cfg.yml_dict:
                logger.info('New key {} ---> {}'.format(key, getattr(cfg, key)))
                cfg.yml_dict[key] = getattr(cfg, key)
            else:
                orig_val = cfg.yml_dict[key]
                new_val = getattr(cfg, key)
                if orig_val != new_val:
                    logger.info('Existing key {} ---> {} from {}'.format(key, new_val, orig_val))
                    cfg.yml_dict[key] = new_val

    from easydict import EasyDict
    def log_config_to_file(cfg, pre='cfg_yml'):
        print("{} Config {} details below {}".format("="*20, pre, "="*20))
        for key, val in cfg.items():
            if isinstance(cfg[key], EasyDict):
                print('--- %s.%s = edict() ---' % (pre, key))
                log_config_to_file(cfg[key], pre=pre + '.' + key)
                continue
            print('%s.%s: %s' % (pre, key, val))
        print("{} Config {} details above {}".format("-"*20, pre, "-"*20))
    
    # log_config_to_file(cfg.yml_dict)
    
    return cfg, logger, None


def init_basics_imle(subset):
    """
    Init the basic configurations for the experiment.
    """

    """Load the config file"""
    
    if subset in ["eth", "univ", "hotel", "zara1", "zara2"]:
        dir_name = 'eth_ucy'
    elif subset == "sdd":
        dir_name = 'sdd'
    else:
        dir_name = 'eth_ucy'
        
    cfg = Config(f'extern/traj/MoFlow/cfg/{dir_name}/imle.yml', 'reproduce')
    cfg.ckpt_path = f'experiments/config/traj/MoFlow/fm/{subset}/ckpt.pt'

    tag = '_'

    cfg.MODEL.USE_PRE_NORM = False
    cfg.latent_tau = 0
    cfg.objective = 'set'
    cfg.num_to_gen = 20
    cfg.load_pretrained = False

    tag += f'_IMLE_gen_{cfg.objective}_M_{cfg.num_to_gen}'

    cfg.loss_reg_gt_weight = 0.0
    cfg.loss_reg_chamfer_weight = 1.0
    tag += '_GT_{:.2f}_Chamfer_{:.2f}'.format(cfg.loss_reg_gt_weight, cfg.loss_reg_chamfer_weight)

    cfg.loss_reg_reduction = 'sum'
    cfg.loss_reg_squared = False

    tag += f'_REG_{cfg.loss_reg_reduction[:1].upper()}'

    cfg.subset = subset
    tag += f'_{cfg.subset}'

    cfg.rotate = True
    if True:
        cfg.rotate_time_frame = 6
        cfg.rotate_aug = False
        tag += f'_rot_{cfg.rotate_time_frame}'
        if cfg.rotate_aug:
            tag += '_aug'

    # change: 'min_max' -> 'original'
    cfg.data_norm = 'original'
    tag += f'_{cfg.data_norm}'

    cfg.OPTIMIZATION.LR = 0.0001

    tag += f'_LR{cfg.OPTIMIZATION.LR}_WD{cfg.OPTIMIZATION.WEIGHT_DECAY}'

    cfg.OPTIMIZATION.NUM_EPOCHS = 50

    cfg.train_batch_size = 24
    cfg.test_batch_size = 24

    cfg.checkpt_freq = 1
    
    cfg.max_num_ckpts = 5

    tag += f'_BS{cfg.train_batch_size}_EP{cfg.OPTIMIZATION.NUM_EPOCHS}'

    tag = tag.replace('__', '_')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = logging.getLogger(__name__)
    
    return cfg, logger, None


class DummyArgs:
    def __init__(self, cfg):
        # self.load_pretrained = True
        self.load_pretrained = False
        self.ckpt_path = cfg.ckpt_path


