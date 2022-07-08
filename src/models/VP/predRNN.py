from pathlib import Path
import numpy as np
from yacs.config import CfgNode
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import optimizer_to_cuda


class ActionCondPredRNN_v2(nn.Module):
    def __init__(self, cfg: CfgNode, load: bool=False) -> None:
        super(ActionCondPredRNN_v2, self).__init__()
        
        self.itr = 0
        
        self.obs_len = cfg.DATA.OBSERVE_LENGTH
        self.pred_len=cfg.DATA.PREDICT_LENGTH
        self.output_path = Path(cfg.OUTPUT_DIR)
        
        if cfg.DATA.DATASET_NAME == "bair":
            self.img_width = 64
            lr = 0.0001
            self.r_sampling_step_1 = 50000
            self.r_sampling_step_2 = 100000
            self.r_exp_alpha = 5000
            self.img_channel = 3
        else:
            raise ValueError
        
        self.patch_size = 1
        num_hidden = [128, 128, 128, 128]
        num_layers = len(num_hidden)
        filter_size = 5
        stride = 1
        layer_norm = 0
        num_action_ch = 4
        decouple_beta = 0.1

        #self.configs = configs
        self.conv_on_input = 1
        self.res_on_conv = 1
        self.patch_height = self.img_width // self.patch_size
        self.patch_width = self.img_width // self.patch_size
        self.patch_ch = self.img_channel * (self.patch_size ** 2)
        self.action_ch = num_action_ch
        self.rnn_height = self.patch_height
        self.rnn_width = self.patch_width
        
        if self.conv_on_input == 1:
            self.rnn_height = self.patch_height // 4
            self.rnn_width = self.patch_width // 4
            self.conv_input1 = nn.Conv2d(self.patch_ch, num_hidden[0] // 2,
                                         filter_size,
                                         stride=2, padding=filter_size // 2, bias=False)
            self.conv_input2 = nn.Conv2d(num_hidden[0] // 2, num_hidden[0], filter_size, stride=2,
                                         padding=filter_size // 2, bias=False)
            self.action_conv_input1 = nn.Conv2d(self.action_ch, num_hidden[0] // 2,
                                                filter_size,
                                                stride=2, padding=filter_size // 2, bias=False)
            self.action_conv_input2 = nn.Conv2d(num_hidden[0] // 2, num_hidden[0], filter_size, stride=2,
                                                padding=filter_size // 2, bias=False)
            self.deconv_output1 = nn.ConvTranspose2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1] // 2,
                                                     filter_size, stride=2, padding=filter_size // 2,
                                                     bias=False)
            self.deconv_output2 = nn.ConvTranspose2d(num_hidden[num_layers - 1] // 2, self.patch_ch,
                                                     filter_size, stride=2, padding=filter_size // 2,
                                                     bias=False)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        self.decouple_beta = decouple_beta
        self.MSE_criterion = nn.MSELoss().cuda()
        self.norm_criterion = nn.SmoothL1Loss().cuda()

        for i in range(num_layers):
            if i == 0:
                in_channel = self.patch_ch + self.action_ch if self.conv_on_input == 0 else num_hidden[0]
            else:
                in_channel = num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell_action(in_channel, num_hidden[i], self.rnn_width,
                                       filter_size, stride, layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        if self.conv_on_input == 0:
            self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.patch_ch + self.action_ch, 1, stride=1,
                                       padding=0, bias=False)
        self.adapter = nn.Conv2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1], 1, stride=1, padding=0,
                                 bias=False)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.optimizers = [self.optimizer]
        
        if load:
            self.load()
        

    def forward(self, frames, mask_true):
        input_frames = frames[:, :, :self.patch_ch, :, :]
        input_actions = frames[:, :, self.patch_ch:, :, :]
        
        batch_size = input_frames.shape[1]

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch_size, self.num_hidden[i], self.rnn_height, self.rnn_width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        decouple_loss = []
        memory = torch.zeros([batch_size, self.num_hidden[0], self.rnn_height, self.rnn_width]).cuda()

        for t in range(self.obs_len + self.pred_len - 1):
            if t == 0:
                net = input_frames[t]
            else:
                net = mask_true[t - 1][:, None, None, None] * input_frames[t] + \
                    (1 - mask_true[t - 1])[:, None, None, None] * x_gen
            action = input_actions[t]

            if self.conv_on_input == 1:
                net_shape1 = net.size()
                net = self.conv_input1(net)
                if self.res_on_conv == 1:
                    input_net1 = net
                net_shape2 = net.size()
                net = self.conv_input2(net)
                if self.res_on_conv == 1:
                    input_net2 = net
                action = self.action_conv_input1(action)
                action = self.action_conv_input2(action)

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory, action)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory, action)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(0, self.num_layers):
                decouple_loss.append(torch.mean(torch.abs(
                    torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))
            if self.conv_on_input == 1:
                if self.res_on_conv == 1:
                    x_gen = self.deconv_output1(h_t[self.num_layers - 1] + input_net2, output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen + input_net1, output_size=net_shape1)
                else:
                    x_gen = self.deconv_output1(h_t[self.num_layers - 1], output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen, output_size=net_shape1)
            else:
                x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
            
        next_frames = torch.stack(next_frames, dim=0)
        return next_frames, decouple_loss
    
    def predict(self, data_dict):
        batch_size = data_dict['obs'].shape[1]
        
        mask_true = self.reserve_schedule_sampling_exp(np.inf, batch_size)
        frames = torch.cat([data_dict['obs'], data_dict['gt']], dim=0)
        if self.patch_size > 1:
            frames = reshape_patch(frames, self.patch_size)
        next_frames, _ = self.forward(frames, mask_true)
        if self.patch_size > 1:
            next_frames = reshape_patch_back(next_frames, self.patch_size)
        data_dict["pred"] = next_frames[-self.pred_len:]
        return data_dict

    def update(self, data_dict):
        self.optimizer.zero_grad()
        
        batch_size = data_dict['obs'].shape[1]
        
        mask_true = self.reserve_schedule_sampling_exp(self.itr, batch_size)
        self.itr += 1
        
        frames = torch.cat([data_dict['obs'], data_dict['gt']], dim=0)
        if self.patch_size > 1:
            frames = reshape_patch(frames, self.patch_size)
        next_frames, decouple_loss = self.forward(frames, mask_true)
        
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        mse_loss = self.MSE_criterion(next_frames, frames[1:, :, :self.img_channel]) 
        loss = mse_loss + self.decouple_beta * decouple_loss
        loss.backward()
        
        self.optimizer.step()
        
        return {"mse": mse_loss.item(), "decouple_loss": decouple_loss.item()}

    def save(self, path: Path=None):
        if path is None:
            path = self.output_path / "ckpt.pt"
            
        ckpt = {
            'model_state': self.state_dict(),
            'model_optim_state': self.optimizer.state_dict(),
        }

        torch.save(ckpt, path)

    def load(self, path: Path=None):
        if path is None:
            path = self.output_path / "ckpt.pt"
        
        ckpt = torch.load(path)

        self.load_state_dict(ckpt['model_state'])
        #self.load_state_dict(ckpt['net_param'])
        try:
            self.optimizer.load_state_dict(ckpt['model_optim_state'])

            optimizer_to_cuda(self.optimizer)
        except KeyError:
            pass
        
    def reserve_schedule_sampling_exp(self, itr, batch_size):
        if itr < self.r_sampling_step_1:
            r_eta = 0.5
        elif itr < self.r_sampling_step_2:
            r_eta = 1.0 - 0.5 * np.exp(-float(itr - self.r_sampling_step_1) / self.r_exp_alpha)
        else:
            r_eta = 1.0

        if itr < self.r_sampling_step_1:
            eta = 0.5
        elif itr < self.r_sampling_step_2:
            eta = 0.5 - (0.5 / (self.r_sampling_step_2 - self.r_sampling_step_1)) * (itr - self.r_sampling_step_1)
        else:
            eta = 0.0

        r_random_flip = np.random.random_sample(
            (self.obs_len - 1, batch_size))
        r_true_token = (r_random_flip < r_eta)

        random_flip = np.random.random_sample(
            (self.pred_len - 1, batch_size))
        true_token = (random_flip < eta)

        return torch.Tensor(np.concatenate([r_true_token, true_token], axis=0)).cuda()
            
    

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m

class SpatioTemporalLSTMCell_action(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell_action, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_a = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_a = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, h_t, c_t, m_t, a_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        a_concat = self.conv_a(a_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat * a_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m

class PredRNN_v2(nn.Module):
    def __init__(self, cfg: CfgNode, load: bool=False) -> None:
        super(PredRNN_v2, self).__init__()
        
        self.itr = 0
        
        self.obs_len = cfg.DATA.OBSERVE_LENGTH
        self.pred_len=cfg.DATA.PREDICT_LENGTH
        self.output_path = Path(cfg.OUTPUT_DIR)

        if cfg.DATA.DATASET_NAME == "kth":
            self.img_width = 128
            lr = 0.0001
            self.decouple_beta = 0.01
            self.r_sampling_step_1 = 10000
            self.r_sampling_step_2 = 100000
            self.r_exp_alpha = 4000
        elif cfg.DATA.DATASET_NAME == "mnist":
            self.img_width = 64
            lr = 0.0001
            self.decouple_beta = 0.1
            self.r_sampling_step_1 = 50000
            self.r_sampling_step_2 = 100000
            self.r_exp_alpha = 5000
        else:
            raise ValueError    
        
        self.patch_size = 4
        self.img_channel = 1
        num_hidden = [128, 128, 128, 128]
        num_layers = len(num_hidden)
        filter_size = 5
        stride = 1
        layer_norm = 0
            
        self.frame_channel = self.patch_size * self.patch_size * self.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = self.img_width // self.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, filter_size,
                                       stride, layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.optimizers = [self.optimizer]
        
        if load:
            self.load()
            

    def forward(self, frames, mask_true):
        batch = frames.shape[1]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).cuda()

        for t in range(self.obs_len + self.pred_len - 1):
            if t == 0:
                net = frames[t]
            else:
                net = mask_true[t - 1][:, None, None, None] * frames[t] + (1 - mask_true[t - 1])[:, None, None, None] * x_gen
            
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))
        
        next_frames = torch.stack(next_frames, dim=0)
            
        return next_frames, decouple_loss

    def predict(self, data_dict):
        batch_size = data_dict['obs'].shape[1]
        
        mask_true = self.reserve_schedule_sampling_exp(np.inf, batch_size)
        frames = torch.cat([data_dict['obs'], data_dict['gt']], dim=0)
        if self.patch_size > 1:
            frames = reshape_patch(frames, self.patch_size)
        next_frames, _ = self.forward(frames, mask_true)
        if self.patch_size > 1:
            next_frames = reshape_patch_back(next_frames, self.patch_size)
        data_dict["pred"] = next_frames[-self.pred_len:]
        return data_dict

    def update(self, data_dict):
        self.optimizer.zero_grad()
        
        batch_size = data_dict['obs'].shape[1]
        
        mask_true = self.reserve_schedule_sampling_exp(self.itr, batch_size)
        self.itr += 1
        frames = torch.cat([data_dict['obs'], data_dict['gt']], dim=0)
        if self.patch_size > 1:
            with torch.no_grad():
                frames = reshape_patch(frames, self.patch_size)
        next_frames, decouple_loss = self.forward(frames, mask_true)
        
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        mse_loss = self.MSE_criterion(next_frames, frames[1:]) 
        loss = mse_loss + self.decouple_beta * decouple_loss
        loss.backward()
        
        self.optimizer.step()
        
        return {"mse": mse_loss.item(), "decouple_loss": decouple_loss.item()}

    def save(self, path: Path=None):
        if path is None:
            path = self.output_path / "ckpt.pt"
            
        ckpt = {
            'model_state': self.state_dict(),
            'model_optim_state': self.optimizer.state_dict(),
        }

        torch.save(ckpt, path)

    def load(self, path: Path=None):
        if path is None:
            path = self.output_path / "ckpt.pt"
        
        #import pdb;pdb.set_trace()
        ckpt = torch.load(path)

        self.load_state_dict(ckpt['model_state'])
        #self.load_state_dict(ckpt['net_param'])
        try:
            self.optimizer.load_state_dict(ckpt['model_optim_state'])

            optimizer_to_cuda(self.optimizer)
        except KeyError:
            pass
        
    def reserve_schedule_sampling_exp(self, itr, batch_size):
        
        if itr < self.r_sampling_step_1:
            r_eta = 0.5
        elif itr < self.r_sampling_step_2:
            r_eta = 1.0 - 0.5 * np.exp(-float(itr - self.r_sampling_step_1) / self.r_exp_alpha)
        else:
            r_eta = 1.0

        if itr < self.r_sampling_step_1:
            eta = 0.5
        elif itr < self.r_sampling_step_2:
            eta = 0.5 - (0.5 / (self.r_sampling_step_2 - self.r_sampling_step_1)) * (itr - self.r_sampling_step_1)
        else:
            eta = 0.0

        r_random_flip = np.random.random_sample(
            (self.obs_len - 1, batch_size))
        r_true_token = (r_random_flip < r_eta)

        random_flip = np.random.random_sample(
            (self.pred_len - 1, batch_size))
        true_token = (random_flip < eta)

        return torch.Tensor(np.concatenate([r_true_token, true_token], axis=0)).cuda()
            

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    seq_length, batch_size, num_channels, img_height, img_width = img_tensor.size()
    a = torch.reshape(img_tensor, [seq_length, batch_size, num_channels,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size])
                                
    b = torch.permute(a, (0,1,2,4,6,3,5))
    patch_tensor = torch.reshape(b, [seq_length, batch_size,
                                     patch_size*patch_size*num_channels,
                                  img_height//patch_size,
                                  img_width//patch_size])
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    seq_length, batch_size, channels, patch_height, patch_width = patch_tensor.size()
    
    img_channels = channels // (patch_size*patch_size)
    a = torch.reshape(patch_tensor, [seq_length, batch_size,
                                     img_channels,
                                     patch_size, patch_size,
                                  patch_height, patch_width])
                                  
    b = torch.permute(a, (0,1,2,5,3,6,4))
    img_tensor = torch.reshape(b, [seq_length, batch_size,
                                   img_channels,
                                patch_height * patch_size,
                                patch_width * patch_size])
    return img_tensor