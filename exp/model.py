import torch
import torchcde
import torch.nn as nn
import torchdiffeq


######################
# Neural CDEs are defined by a function which takes in a time and a hidden state, and outputs a new hidden state.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)

        z = z.tanh()

        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


class NeuralCDE(torch.nn.Module):
    def __init__(
        self, input_channels, hidden_channels, output_channels, interpolation="cubic"
    ):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == "cubic":
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == "linear":
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError(
                "Only 'linear' and 'cubic' interpolation methods are implemented."
            )

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.interval)

        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y


######################
# Neural latent ODE
######################
class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


# from hidden state get output
class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar))


# 正态分布的KL散度
def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.0
    lstd2 = lv2 / 2.0

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2)) - 0.5
    return kl


class LatentODE(nn.Module):
    def __init__(self, params):
        super(LatentODE, self).__init__()

        self.adjoint = params["adjoint"]

        self.latent_dim = params["latent_dim"]
        self.nhidden = params["nhidden"]
        self.obs_dim = params["obs_dim"]
        self.rnn_nhidden = params["rnn_nhidden"]
        self.nspiral = params["nspiral"]

        self.rec = RecognitionRNN(
            latent_dim=self.latent_dim,
            obs_dim=self.obs_dim,
            nhidden=self.rnn_nhidden,
            nbatch=self.nspiral,
        )
        self.dec = Decoder(
            latent_dim=self.latent_dim, obs_dim=self.obs_dim, nhidden=self.nhidden
        )
        self.func = LatentODEfunc(latent_dim=self.latent_dim, nhidden=self.nhidden)

        if self.adjoint:
            self.odeint = torchdiffeq.odeint_adjoint
        else:
            self.odeint = torchdiffeq.odeint

    def forward(self, x, t):
        h = self.rec.initHidden().to(x.device)
        for t in reversed(x.size(1)):
            obs = x[:, t, :]
            out, h = self.rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, : self.latent_dim], out[:, self.latent_dim :]
        epsilon = torch.randn(qz0_mean.size()).to(x.device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean

        pred_z = self.odeint(self.func, z0, t).permute(1, 0, 2)
        pred_x = self.dec(pred_z)
