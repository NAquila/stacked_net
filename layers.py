"""
This module contains all the stacked layers.
"""
import torch
from torch import nn
from torch.nn.parameter import Parameter

from .misc import stacked_batch_norm, AdditiveNoise


class StackedModule(nn.Module):
    """
    The simplest stacked module.

    Is not stacked at all but just contains a list of all modules and
    passes each layer of the stacked input to the correct module.
    Slow computation.

    ### Parameters
    - *m_list*: list(nn.Module), required
    + List of modules to stack.
    """

    def __init__(self, m_list):
        super(StackedModule, self).__init__()
        self.m_list = nn.ModuleList(m_list)

    def forward(self, x):
        out = [self.m_list[i](x[i]) for i in range(len(self.m_list))]
        return torch.stack(out)

    def to_list(self):
        return list(self.m_list)


class StackedLinear(nn.Module):
    """
    Stacked version of the nn.Linear.

    Works by leveraging torch.matmul and other tensor functions.
    Quick computation.

    ### Parameters
    - *m_list*, list(nn.Linear), required
    + List of networks to stack. These all need to be of equal shape.
    """

    def __init__(self, m_list):
        super(StackedLinear, self).__init__()

        w_list = []
        wg_list = []
        for m in m_list:
            w_list.append(m.weight.data)
            if m.weight.grad is None:
                wg_list.append(torch.zeros_like(m.weight))
            else:
                wg_list.append(m.weight.grad.data)

        self.weight = Parameter(torch.stack(w_list))
        self.weight.grad = torch.stack(wg_list)
        if m_list[0].bias is None:
            self.register_parameter('bias', None)
        else:

            b_list = []
            bg_list = []
            for m in m_list:
                # Needs to be unsqueezed so that broadcasting will work
                b_list.append(m.bias.data.unsqueeze(0))
                if m.bias.grad is None:
                    bg_list.append(torch.zeros_like(m.bias).unsqueeze(0))
                else:
                    bg_list.append(m.bias.grad.data.unsqueeze(0))

            self.bias = Parameter(torch.stack(b_list))
            self.bias.grad = torch.stack(bg_list)

    def forward(self, x):
        out = x.matmul(self.weight.transpose(1, 2))
        if self.bias is not None:
            out += self.bias
        return out

    def to_list(self):
        """ Return a list of nn.Linear."""
        lin_list = []
        num_nets, out_features, in_features = self.weight.shape
        bias = self.bias is not None
        for i in range(num_nets):
            lin = nn.Linear(in_features, out_features, bias=bias)
            lin.weight = Parameter(self.weight[i])
            lin.weight.grad = self.weight[i].grad
            if bias:
                lin.bias = Parameter(self.bias[i].squeeze())
                if self.bias[i].grad is not None:
                    lin.bias.grad = self.bias[i].grad.squeeze()
            lin_list.append(lin)
        return lin_list


class StackedReLU(nn.Module):
    """
    Stacked version of the nn.ReLU loss layer.

    Since loss functions broadcasts, just uses the first module
    in calculations.
    Quick calculations.

    ### Parameters
    - *m_list*: list(nn.ReLU), required
    + List of modules to stack
    """

    def __init__(self, m_list):
        super(StackedReLU, self).__init__()
        self.m_list = m_list

    def forward(self, x):
        return self.m_list[0](x)

    def to_list(self):
        return self.m_list


class StackedELU(nn.Module):
    """
    Stacked version of the nn.ELU loss layer.
    Since loss functions broadcasts, just uses the first module
    in calculations.
    Quick calculations.

    ### Parameters
    - *m_list*: list(nn.ELU), required
    + List of modules to stack
    """

    def __init__(self, m_list):
        super(StackedELU, self).__init__()
        self.m_list = m_list

    def forward(self, x):
        return self.m_list[0](x)

    def to_list(self):
        return self.m_list


class StackedDropout(nn.Module):
    """
    Stacked version of the nn.Dropout layer.

    Implements dropout on each layer.
    Different layers can have different dropout parameters.
    Quick calculations.

    ### Parameters
    - *m_list*: list(nn.Dropout), required
    + List of modules to stack
    """

    def __init__(self, m_list):
        super(StackedDropout, self).__init__()

        self.register_buffer('ip_tensor',
                             torch.Tensor([1-m.p for m in m_list]))
        self.register_buffer('scale', 1/(self.ip_tensor))

    def forward(self, x):
        if self.training:
            ip = self.ip_tensor
            s = self.scale
            # Set up s and ip so they can be broadcasted
            for i in range(len(x.shape)-1):
                ip = ip.unsqueeze(dim=-1)
                s = s.unsqueeze(dim=-1)
            return x * ip.expand(x.shape).bernoulli() * s.expand(x.shape)
        else:
            return x

    def to_list(self):
        return[nn.Dropout(1-ip) for ip in self.ip_tensor]


class StackedAdditiveNoise(nn.Module):
    """
    Stacked version of the AdditiveNoise layer.

    Applies Gaussian noise to each layer during training and does nothing
    at evaluation.
    Different layers can have different sigma parameters.
    Quick calculations.

    ### Parameters
    - *m_list*: list(torch_models.modules.AdditiveNoise), required
    + List of modules to stack
    """

    def __init__(self, m_list):
        super(StackedAdditiveNoise, self).__init__()
        self.register_buffer('sigma_tensor',
                             torch.Tensor([m.sigma for m in m_list]))

    def forward(self, x):
        if self.training:
            s = self.sigma_tensor
            for i in range(len(x.shape)-1):
                s = s.unsqueeze(dim=-1)
            noise = s * torch.randn_like(x)
            return x + noise
        else:
            return x

    def to_list(self):
        return[AdditiveNoise(sigma) for sigma in self.sigma_tensor]


class StackedBatchNorm1D(nn.Module):
    """
    Stacked implementation of the nn.BatchNorm1d layer.

    Requires all nets to have 'num_features', 'affine',
    'track_running_stats' and 'training' set equal.

   ### Parameters
    - *m_list*: list(nn.BatchNorm1d), required
    + List of modules to stack
    """
    def __init__(self, m_list):

        super(StackedBatchNorm1D, self).__init__()
        self.num_features = m_list[0].num_features
        assert all(self.num_features == m.num_features for m in m_list)
        self.register_buffer('eps_tensor',
                             torch.tensor([m.eps for m in m_list],
                                          dtype=torch.float32))
        self.register_buffer('momentum_tensor',
                             torch.tensor([m.momentum for m in m_list],
                                          dtype=torch.float32))
        self.affine = m_list[0].affine
        assert all(self.affine == m.affine for m in m_list)
        self.track_running_stats = m_list[0].track_running_stats
        assert all(self.track_running_stats == m.track_running_stats
                   for m in m_list)
        self.num_nets = len(m_list)

        if self.affine:

            self.weight = Parameter(torch.Tensor(self.num_nets, 1,
                                                 self.num_features),
                                    requires_grad=True)

            self.weight.grad = torch.zeros(self.num_nets, 1,
                                           self.num_features)

            self.bias = Parameter(torch.Tensor(self.num_nets, 1,
                                               self.num_features),
                                  requires_grad=True)

            self.bias.grad = torch.zeros(self.num_nets, 1,
                                         self.num_features)

            for i, m in enumerate(m_list):
                self.weight.data[i] = m.weight.data
                if m.weight.grad is not None:
                    self.weight.grad[i] = m.weight.grad
                self.bias.data[i] = m.bias.data
                if m.bias.grad is not None:
                    self.bias.grad[i] = m.bias.grad
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean',
                                 torch.zeros(self.num_nets, 1,
                                             self.num_features))

            self.register_buffer('running_var',
                                 torch.ones(self.num_nets, 1,
                                            self.num_features))

            self.register_buffer('num_batches_tracked',
                                 torch.zeros([self.num_nets, 1],
                                             dtype=torch.long))

            for i, m in enumerate(m_list):
                self.running_mean[i] = m.running_mean
                self.running_var[i] = m.running_var
                self.num_batches_tracked[i] = m.num_batches_tracked
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            exponential_average_factor = self.momentum_tensor

        output = stacked_batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps_tensor)

        # output, self.running_mean, self.running_var = tmp
        return output

    def to_list(self):
        m_list = [torch.nn.BatchNorm1d(self.num_features,
                                       eps, mom, self.affine,
                                       self.track_running_stats)
                  for eps, mom in zip(self.eps_tensor, self.momentum_tensor)]

        for i, m in enumerate(m_list):
            if self.track_running_stats:
                m.running_mean.data = (self.running_mean.data[i, :, :]
                                       .squeeze(0))
                if self.running_mean.grad is not None:
                    m.running_mean.grad = (self.running_mean.grad[i, :, :]
                                           .squeeze(0))
                m.running_var.data = self.running_var.data[i, :, :].squeeze(0)
                if self.running_var.grad is not None:
                    m.running_var.grad = (self.running_var.grad[i, :, :]
                                          .squeeze(0))
                m.num_batches_tracked.data = (self.num_batches_tracked
                                              .data[i, 0])

            if self.affine:
                m.weight.data = self.weight.data[i, :, :].squeeze(0)
                m.bias.data = self.bias.data[i, :, :].squeeze(0)

        return m_list
