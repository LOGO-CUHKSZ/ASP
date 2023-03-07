"Implementation is from https://github.com/karpathy/pytorch-normalizing-flows"

import torch
from torch import nn


class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True, norm=False):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)

        self.norm = norm

    def forward(self, x):
        if self.norm:
            assert ((x < 1).all() and (x >= 0).all()), 'input value should within [0,1)'
            x = torch.log(x / (1 - x))
            extra_log_term = torch.sum(torch.log(1 + 1/(x[:, 1::2]*(1-x[:, 1::2])))-torch.log(x[:, ::2]*(1-x[:, ::2])),dim=1)
        x0, x1 = x[:, ::2], x[:, 1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)

        z0 = x0  # untouched half
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        if self.norm:
            log_det = log_det + extra_log_term
        return z, log_det

    def backward(self, z):
        z0, z1 = z[:, ::2], z[:, 1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        # restrict to [0,1]
        x0 = z0  # this was the same
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        # if self.norm:
        #     x0 = torch.sigmoid(x0)  # this was the same
        #     x1 = torch.sigmoid(x1)
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        # if self.norm:
        #     log_det = torch.sum(-s + torch.log(x1 * (1 - x1)), dim=1)
        x = (x-x.min())/(x.max()-x.min())
        log_det = log_det - z.shape[1]*torch.log(x.max()-x.min())
        return x, log_det


class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None

    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = torch.log(x/(1-x)) * torch.exp(s) + t
        log_det = torch.sum(s-torch.log(torch.sigmoid(x))-torch.log(1-torch.sigmoid(x)), dim=1)
        return z, log_det

    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = torch.sigmoid((z - t) * torch.exp(-s))
        log_det = torch.sum(-s-torch.log(x)-torch.log(1-x), dim=1)
        return x, log_det



# ------------------------------------------------------------------------

class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m).to(z.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)

    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs
