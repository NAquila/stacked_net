"""
Miscellaneous stuff goes here
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import MSELoss
from ignite.metrics.metric import Metric
import logging
from enum import Enum


class EarlyStoppingEvents(Enum):
    EARLY_STOPPING_DONE = "early_stopping_done"
    HAS_IMPROVED = "has_improved"
    NET_CONVERGED = "net_converged"


class KindEarlyStopping(object):
    """EarlyStopping handler can be used to stop the training if no improvement
    after a given number of events.
    Args:
        patience (int):
            Number of events to wait if no improvement
    and then stop the training
        score_function (Callable):
            It should be a function taking a single argument,
    an `ignite.engine.Engine` object,
            and return a score `float`.
    An improvement is considered if the score is higher.
        trainer (Engine):
            trainer engine to stop the run if no improvement
    """
    def __init__(self, patience, score_function):

        if not callable(score_function):
            raise TypeError("Argument score_function should be a function")

        if patience < 1:
            raise ValueError("Argument patience should be positive integer")

        self.score_function = score_function
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self._logger = logging.getLogger(__name__ + "."
                                         + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())

    def __call__(self, engine):
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            self._logger.debug("%i / %i" % (self.counter, self.patience))
            if self.counter >= self.patience:
                msg = "Training can be stopped!"
                self._logger.info(msg)
                engine.fire_event(EarlyStoppingEvents.NET_CONVERGED)
                engine.fire_event(EarlyStoppingEvents.EARLY_STOPPING_DONE)

        else:
            if self.counter >= self.patience:
                msg = ("Achieved additional improvement"
                       " after {} checks, this is ignored"
                       .format(self.counter))
                self._logger.info(msg)

            self.best_score = score
            self.counter = 0
            engine.fire_event(EarlyStoppingEvents.HAS_IMPROVED)


class StackedMSELoss(MSELoss):
    """
    Wrapper used around MSELoss to get the MSE_loss for each network.
    ### Parameters:
    - *num_nets* (int)
    """

    def __init__(self, size_average=None, reduce=None,
                 reduction='elementwise_mean'):
        super(StackedMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        if self.reduction == 'stacked_elementwise_mean':
            loss = F.mse_loss(input, target, reduction='none')
            loss = loss.mean(-1).mean(-1)
        elif self.reduction == 'elementwise_mean':
            loss = F.mse_loss(input, target, reduction=self.reduction)
            loss = loss*input.size(0)
        else:
            loss = F.mse_loss(input, target, reduction=self.reduction)
        return loss


class StackedEarlyStopping(object):
    """EarlyStopping handler can be used to stop
    the training if no improvement after a given number of events
    Args:
        patience (int):
            Number of events to wait if no improvement
    and then stop the training
        score_function (Callable):
            It should be a function taking a single argument,
    an `ignite.engine.Engine` object,
            and return a score `float`.
    An improvement is considered if the score is higher.
        trainer (Engine):
            trainer engine to stop the run if no improvement
    """
    def __init__(self, patience, score_function, num_nets):

        if not callable(score_function):
            raise TypeError("Argument score_function should be a function")

        if patience < 1:
            raise ValueError("Argument patience should be positive integer")

        self.num_nets = num_nets
        self.score_function = score_function
        self.patience = patience
        self.counter_list = [0] * num_nets
        self.best_score_list = [None] * num_nets
        self.done_list = [False] * num_nets
        self._logger = logging.getLogger(__name__ + "."
                                         + self.__class__.__name__)
        self._logger.addHandler(logging.NullHandler())

    def __call__(self, engine):
        score_vec = self.score_function(engine)

        for i in range(self.num_nets):
            if self.best_score_list[i] is None:
                self.best_score_list[i] = score_vec[i]
            elif score_vec[i] < self.best_score_list[i]:
                self.counter_list[i] += 1
                self._logger.debug("%i / %i"
                                   % (self.counter_list[i], self.patience))
                if self.counter_list[i] >= self.patience:
                    msg = "Net {}, training can be stopped!".format(i)
                    self._logger.info(msg)
                    if not self.done_list[i]:
                        engine.fire_event(EarlyStoppingEvents.NET_CONVERGED)
                        self.done_list[i] = True

            else:
                if score_vec[i] < self.best_score_list[i]:
                    msg = ("Achieved additional improvement for net {}"
                           " after {} checks, this is ignored"
                           .format(i, self.counter_list[i]))
                    self._logger.info(msg)

                self.best_score_list[i] = score_vec[i]
                self.counter_list[i] = 0
                engine.fire_event(EarlyStoppingEvents.HAS_IMPROVED)

        if all(self.done_list):
            engine.fire_event(EarlyStoppingEvents.EARLY_STOPPING_DONE)


class StackedLossMetric(Metric):
    """
    Calculates the average loss according to the passed loss_fn,
    in the format it was provided.
    """

    def __init__(self, loss_fn, output_transform=lambda x: x,
                 batch_size=lambda x: x.shape[1]):
        super(StackedLossMetric, self).__init__(output_transform)
        self._loss_fn = loss_fn
        self._batch_size = batch_size

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        if len(output) == 2:
            y_pred, y = output
            kwargs = {}
        else:
            y_pred, y, kwargs = output
        average_loss = self._loss_fn(y_pred, y, **kwargs)

        N = self._batch_size(y)
        self._sum += average_loss * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise Exception(
                'Loss must have at least one'
                + ' example before it can be computed')
        return self._sum / self._num_examples


def stacked_batch_norm(input, running_mean, running_var,
                       weight=None, bias=None,
                       training=False, momentum_tensor=torch.tensor([0.1]),
                       eps_tensor=torch.tensor([1e-5])):
    """
    Applies Batch Normalization along the first dimension of the input.

    ### Parameters:
    - *input* (torch.tensor): At least 2d, e.g. (M, N, L)
    - *running_mean* (torch.tensor): e.g. (M, 1, L)
    - *running_var* (torch.tensor): e.g. (M, 1, L)
    - *weight* (torch.tensor): e.g. (M, 1, L)
    - *bias* (torch.tensor): e.g. (M, 1, L)
    - *training* (bool)
    - *momentum_tensor*
    - *eps_tensor*
    """

    if training:
        # Scale with the means and variances of the batch.
        # To be consistent with pytorch's batchnorm
        # the variance is not bias-corrected
        # with input - (M, N, L) these are (M, 1, L)
        mean_tensor = input.mean(1).unsqueeze(1)
        var_tensor = input.var(1, unbiased=False).unsqueeze(1)
        output = ((input - mean_tensor) /
                  (var_tensor + eps_tensor
                   .reshape(eps_tensor.size(0), 1, 1)).sqrt())

        # Tracing gradients on running_mean
        # or running_var will cause memory leak
        with torch.no_grad():
            # Update moving averages inplace
            if running_mean is not None:
                (running_mean.mul_(1-momentum_tensor
                                   .reshape(momentum_tensor.size(0), 1, 1))
                 .add_(momentum_tensor.reshape(momentum_tensor.size(0), 1, 1)
                       * mean_tensor))

            if running_var is not None:
                (running_var.mul_(1-momentum_tensor
                                  .reshape(momentum_tensor.size(0), 1, 1))
                 .add_(momentum_tensor.reshape(momentum_tensor.size(0), 1, 1)
                       * var_tensor))

    else:
        output = ((input - running_mean) /
                  (running_var + eps_tensor
                   .reshape(eps_tensor.size(0), 1, 1)).sqrt())

    # Rescale
    if weight is not None:
        output = output * weight
    if bias is not None:
        output = output + bias

    return output


class AdditiveNoise(nn.Module):
    """
    Layer that adds Gaussian noise during training and does nothing
    during evaluation.
    """
    def __init__(self, sigma=0.01):
        """
        ### Parameters
        - *sigma*, optional, default=0.01
        + Standard deviation of the gaussian noise
        """

        super(AdditiveNoise, self).__init__()
        self.sigma = sigma
        # Quick fix
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

    def forward(self, x):
        if self.training:
            noise = self.sigma * torch.randn(x.shape, device=self.device)
            return x + noise
        else:
            return x
