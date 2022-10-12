from pathlib import Path
from yacs.config import CfgNode
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import optimizer_to_cuda


class SVG(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super(SVG, self).__init__()

        self.obs_len = cfg.DATA.OBSERVE_LENGTH
        self.pred_len=cfg.DATA.PREDICT_LENGTH
        self.output_path = Path(cfg.OUTPUT_DIR)

        self.g_dim = 128
        if cfg.DATA.DATASET_NAME == "mnist":
            self.z_dim = 10
            self.beta = 0.0001
            self.channels = 1
            self.lr = 0.002
            self.encoder = dcgan_encoder(self.g_dim, self.channels)
            self.decoder = dcgan_decoder(self.g_dim, self.channels)
        elif cfg.DATA.DATASET_NAME == "bair":
            self.z_dim = 64
            self.beta = 0.0001
            self.channels = 7
            self.lr = 0.002
            self.encoder = vgg_encoder(self.g_dim, self.channels)
            self.decoder = vgg_decoder(self.g_dim, self.channels)
        elif cfg.DATA.DATASET_NAME == "kth":
            self.z_dim = 24
            self.beta = 0.000001
            self.channels = 1
            self.lr = 0.0008
            self.encoder = vgg_encoder(self.g_dim, self.channels)
            self.decoder = vgg_decoder(self.g_dim, self.channels)        
        else:
            raise(ValueError)


        self.rnn_size = 256
        self.prior_rnn_layers = 1
        self.posterior_rnn_layers = 1
        self.predictor_rnn_layers = 2

        self.prior = gaussian_lstm(self.g_dim, self.z_dim, self.rnn_size, self.prior_rnn_layers)
        self.posterior = gaussian_lstm(self.g_dim, self.z_dim, self.rnn_size, self.posterior_rnn_layers)
        self.frame_predictor = lstm(self.g_dim+self.z_dim, self.g_dim, self.rnn_size, self.predictor_rnn_layers)

        self.prior.apply(init_weights)
        self.posterior.apply(init_weights)
        self.frame_predictor.apply(init_weights)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

        beta1 = 0.9
        optimizer = torch.optim.Adam
        self.prior_optimizer = optimizer(self.prior.parameters(), lr=self.lr, betas=(beta1, 0.999))
        self.posterior_optimizer = optimizer(self.posterior.parameters(), lr=self.lr, betas=(beta1, 0.999))
        self.frame_predictor_optimizer = optimizer(self.frame_predictor.parameters(), lr=self.lr, betas=(beta1, 0.999))
        self.encoder_optimizer = optimizer(self.encoder.parameters(), lr=self.lr, betas=(beta1, 0.999))
        self.decoder_optimizer = optimizer(self.decoder.parameters(), lr=self.lr, betas=(beta1, 0.999))

        self.optimizers = [self.prior_optimizer, self.posterior_optimizer,
                           self.frame_predictor_optimizer,
                           self.encoder_optimizer, self.decoder_optimizer]

        self.mse_loss = nn.MSELoss()

        self.last_frame_skip = False

    def update(self, data_dict):
        self.frame_predictor.zero_grad()
        self.posterior.zero_grad()
        self.prior.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        
        # initialize the hidden state.

        x = torch.cat([data_dict['obs'], data_dict['gt']], dim=0)

        _, bs, _, _, _ = data_dict['obs'].shape
    
        self.prior.hidden = self.prior.init_hidden(bs)
        self.posterior.hidden = self.posterior.init_hidden(bs)
        self.frame_predictor.hidden = self.frame_predictor.init_hidden(bs)

        mse = 0
        kld = 0
        for i in range(1, self.obs_len+self.pred_len):
            h = self.encoder(x[i-1])
            h_target = self.encoder(x[i])[0]
            if self.last_frame_skip or i < self.obs_len:	
                h, skip = h
            else:
                h = h[0]
            z_t, mu, logvar = self.posterior(h_target)
            _, mu_p, logvar_p = self.prior(h)
            h_pred = self.frame_predictor(torch.cat([h, z_t], 1))
            x_pred = self.decoder([h_pred, skip])
            mse += self.mse_loss(x_pred, x[i])
            kld += kl_criterion(mu, logvar, mu_p, logvar_p)

        loss = mse + kld*self.beta
        loss.backward()

        self.frame_predictor_optimizer.step()
        self.posterior_optimizer.step()
        self.prior_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return {"mse": mse.item(), "kld": kld.item()}
    
    def predict(self, data_dict):
        x = torch.cat([data_dict['obs'], data_dict['gt']], dim=0)
        
        gen_seq = []

        _, bs, _, _, _ = data_dict['obs'].shape
    
        self.prior.hidden = self.prior.init_hidden(bs)
        self.posterior.hidden = self.posterior.init_hidden(bs)
        self.frame_predictor.hidden = self.frame_predictor.init_hidden(bs)
        
        x_in = x[0]
        for i in range(1, self.obs_len+self.pred_len):
            h = self.encoder(x_in)
            if self.last_frame_skip or i < self.obs_len:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < self.obs_len:
                h_target = self.encoder(x[i])[0].detach()
                z_t, _, _ = self.posterior(h_target)
                self.prior(h)
                self.frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
            else:
                z_t, _, _ = self.prior(h)
                h = self.frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = self.decoder([h, skip]).detach()
                gen_seq.append(x_in)
                
        data_dict["pred"] = torch.stack(gen_seq)
        
        return data_dict

    def save(self, epoch: int = 0, path: Path=None) -> None:
        if path is None:
            path = self.output_path / "ckpt.pt"
            
        ckpt = {
            'epoch': epoch,
            'frame_predictor_state': self.frame_predictor.state_dict(),
            'prior_state': self.prior.state_dict(),
            'posterior_state': self.posterior.state_dict(),
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'frame_predictor_optim_state': self.frame_predictor_optimizer.state_dict(),
            'prior_optim_state': self.prior_optimizer.state_dict(),
            'posterior_optim_state': self.posterior_optimizer.state_dict(),
            'encoder_optim_state': self.encoder_optimizer.state_dict(),
            'decoder_optim_state': self.decoder_optimizer.state_dict(),
        }

        torch.save(ckpt, path / "ckpt.pt")
        
    def check_saved_path(self, path: Path = None) -> bool:
        if path is None:
            path = self.output_path / "ckpt.pt"        
        
        return path.exists()

    def load(self, path: Path=None) -> int:
        if path is None:
            path = self.output_path / "ckpt.pt"
        
        ckpt = torch.load(path)

        self.frame_predictor.load_state_dict(ckpt['frame_predictor_state'])
        self.posterior.load_state_dict(ckpt['prior_state'])
        self.prior.load_state_dict(ckpt['posterior_state'])
        self.encoder.load_state_dict(ckpt['encoder_state'], strict=False)
        self.decoder.load_state_dict(ckpt['decoder_state'], strict=False)
        try:
            self.frame_predictor_optimizer.load_state_dict(ckpt['frame_predictor_optim_state'])
            self.posterior_optimizer.load_state_dict(ckpt['posterior_optim_state'])
            self.prior_optimizer.load_state_dict(ckpt['prior_optim_state'])
            self.encoder_optimizer.load_state_dict(ckpt['encoder_optim_state'])
            self.decoder_optimizer.load_state_dict(ckpt['decoder_optim_state'])

            optimizer_to_cuda(self.frame_predictor_optimizer)
            optimizer_to_cuda(self.posterior_optimizer)
            optimizer_to_cuda(self.prior_optimizer)
            optimizer_to_cuda(self.encoder_optimizer)
            optimizer_to_cuda(self.decoder_optimizer)
        except KeyError:
            pass
        
        return ckpt['epoch']

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        #self.hidden = self.init_hidden()

    def init_hidden(self, batch_size):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(batch_size, self.hidden_size).cuda())))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)

class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        #self.hidden = self.init_hidden()

    def init_hidden(self, batch_size):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(batch_size, self.hidden_size).cuda())))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

            
# vgg 64
class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input):
        return self.main(input)

class vgg_encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(vgg_encoder, self).__init__()
        self.dim = dim
        # 64 x 64
        self.c1 = nn.Sequential(
                vgg_layer(nc, 64),
                vgg_layer(64, 64),
                )
        # 32 x 32
        self.c2 = nn.Sequential(
                vgg_layer(64, 128),
                vgg_layer(128, 128),
                )
        # 16 x 16 
        self.c3 = nn.Sequential(
                vgg_layer(128, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 256),
                )
        # 8 x 8
        self.c4 = nn.Sequential(
                vgg_layer(256, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                )
        # 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h1 = self.c1(input) # 64 -> 32
        h2 = self.c2(self.mp(h1)) # 32 -> 16
        h3 = self.c3(self.mp(h2)) # 16 -> 8
        h4 = self.c4(self.mp(h3)) # 8 -> 4
        h5 = self.c5(self.mp(h4)) # 4 -> 1
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class vgg_decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(vgg_decoder, self).__init__()
        self.dim = dim
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = nn.Sequential(
                vgg_layer(512*2, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 256)
                )
        # 16 x 16
        self.upc3 = nn.Sequential(
                vgg_layer(256*2, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 128)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                vgg_layer(128*2, 128),
                vgg_layer(128, 64)
                )
        # 64 x 64
        self.upc5 = nn.Sequential(
                vgg_layer(64*2, 64),
                nn.ConvTranspose2d(64, nc, 3, 1, 1),
                nn.Sigmoid()
                )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec, skip = input 
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 4
        up1 = self.up(d1) # 4 -> 8
        d2 = self.upc2(torch.cat([up1, skip[3]], 1)) # 8 x 8
        up2 = self.up(d2) # 8 -> 16 
        d3 = self.upc3(torch.cat([up2, skip[2]], 1)) # 16 x 16 
        up3 = self.up(d3) # 8 -> 32 
        d4 = self.upc4(torch.cat([up3, skip[1]], 1)) # 32 x 32
        up4 = self.up(d4) # 32 -> 64
        output = self.upc5(torch.cat([up4, skip[0]], 1)) # 64 x 64
        return output



# dcgan 64
class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(dcgan_encoder, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class dcgan_decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(dcgan_decoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        vec, skip = input 
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1, skip[3]], 1))
        d3 = self.upc3(torch.cat([d2, skip[2]], 1))
        d4 = self.upc4(torch.cat([d3, skip[1]], 1))
        output = self.upc5(torch.cat([d4, skip[0]], 1))
        return output


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def kl_criterion(mu1, logvar1, mu2, logvar2):
    sigma1 = logvar1.mul(0.5).exp() 
    sigma2 = logvar2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.mean()
