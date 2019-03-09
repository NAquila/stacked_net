"""
This module contains the main stacked net.
"""
import torch
from torch import nn
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator, Events
from ignite.contrib.handlers import ProgressBar
from ignite import metrics
from copy import deepcopy
from time import time
from enum import Enum
import logging

from . import layers
from .data import StackedTensorDataset, StackedDataLoader
from .misc import StackedMSELoss, StackedLossMetric
from .misc import StackedEarlyStopping, EarlyStoppingEvents
from ..layers import AdditiveNoise


class StackedNet(nn.Module):
    """
    A module that stacks several (small) networks to allow for rapid training.

    Has both a list of single nets and a sequential, stacked net.
    These are not connected but one can update the other.
    All calculations are done on the stacked net.

    ### Parameters
    - *net_list*: list(nn.Sequential), required
    + The networks to stack must all be of feedforward-structure
    with only one level of childrens.
    + The networks must have the same types of
    layers with the same number of nodes.
    """

    def __init__(self, net_list):
        super(StackedNet, self).__init__()
        try:
            assert all(len(net_list[0]) == len(net) for net in net_list)
        except AssertionError:
            raise ValueError("""StackedNet received a list of networks of different lengths.
            This is possibly due to the inclusion of BatchNorm-layers.""")

        params = [param for net in net_list for param in net.parameters()]
        self.device = params[0].device
        try:
            assert all(self.device == p.device for p in params)
        except AssertionError:
            raise ValueError("""StackedNet received a list of networks
            on different devices.
            Please make sure all networks are on the same
            device before initializing StackedNet.""")

        # When changing our nets internal list we shouldn't change the net_list
        # it was created with. Hence we create a new copy here
        self.net_list = deepcopy(net_list)
        self.num_nets = len(net_list)

        self.seq = nn.Sequential()  # Internal structure to propagate data
        self.update_stack(net_list)

    def update_stack(self, net_list):
        """
        This function will create a stack from the given list and assign
        this as the class's sequential net.
        """

        self.seq = nn.Sequential()
        num_modules = len(net_list[0])

        for i in range(num_modules):
            m_list = []
            for j in range(self.num_nets):
                # group together the modules from all nets
                m_list.append(net_list[j][i])
            # Stacked together these modules
            if isinstance(m_list[0], nn.Linear):
                # StackedLinear is a custom Module that concatenates several
                # linear models on top of each other
                m = layers.StackedLinear(m_list)
                suffix = '_lin'
            elif isinstance(m_list[0], nn.ReLU):
                m = layers.StackedReLU(m_list)
                suffix = '_relu'
            elif isinstance(m_list[0], nn.ELU):
                m = layers.StackedELU(m_list)
                suffix = '_elu'
            elif isinstance(m_list[0], nn.Dropout):
                # Optimized dropout
                m = layers.StackedDropout(m_list)
                suffix = '_dropout'
            elif isinstance(m_list[0], nn.BatchNorm1d):
                m = layers.StackedBatchNorm1D(m_list)
                suffix = '_batchnorm'
            elif isinstance(m_list[0], AdditiveNoise):
                m = layers.StackedAdditiveNoise(m_list)
                suffix = '_addnoise'
            else:
                # This is just a list of the modules
                m = layers.StackedModule(m_list)
                suffix = '_other'
            name = str(i) + suffix
            self.seq.add_module(name, m)

        self.seq.to(device=self.device)

        # Set the mode of the model to be the same as that of the
        # first one in the list
        if net_list[0].training:
            self.train()
        else:
            self.eval()

    def forward(self, x):
        """
        Passes stacked input data through the stacked net.
        """

        # All specific forwarding is defined in the submodules
        for name, m in self.seq._modules.items():
            x = m(x)

        return x

    def update_list(self):
        """Updates the list of nets from the internal stacked net"""

        for i, m_stack in enumerate(self.seq):
            m_list = m_stack.to_list()

            for j, m in enumerate(m_list):
                if self.training:
                    m.train()
                else:
                    m.eval()
                # Assign the ith child of the jth net to the new module
                self.net_list[j][i] = m

    def get_nets(self, update=True):
        """
        Return the stacked nets as a net_list.

        ### Parameters
        - *update*: bool, optional
        + Whether to update the nets before or not.
        """

        if update:
            self.update_list()
        return list(self.net_list)

    def fit(self, train_data, val_data, optimizer, batch_size,
            max_epochs, early_stopping=False, check_every=100,
            patience=10, writer=None):
        """
        Function used to fit the stacked autoencoder on data.

        MSELoss used as lossfunction.

        ### Parameters
        - *train_data*: torch.Tensor, required
        + Dataset with the data to train on where samples are in rows
        - *val_data*: torch.Tensor, required
        + Dataset with the data to validate on where samples are in rows
        - *optimizer*: torch.optim.Optimizer, required
        - *batch_size*: int, required
        - *max_epochs*: int, required
        + Training is at most run for this number of epochs
        - *early_stopping*: bool, required
        + Whether to use early stopping or not
        - *check_every*: int, optional, default=1500
        + Scores are calculated every *check_every* iteration.
        - *patience*: int, optional, default=10
        + If the validation score has not increased after *patience* checks
        and early stopping is True, training is stopped.

        ### Returns
        - *time*: Time taken for fitting
        """

        return _stacked_ae_fit(self, train_data=train_data, val_data=val_data,
                               optimizer=optimizer, batch_size=batch_size,
                               max_epochs=max_epochs,
                               early_stopping=early_stopping,
                               check_every=check_every,
                               patience=patience, writer=writer)


def _stacked_ae_fit(stacked_ae, train_data, val_data, batch_size,
                    optimizer, max_epochs, early_stopping=False,
                    check_every=100, patience=10, writer=None):
    """
    Function used to fit a stacked autoencoder on data.

    MSELoss used as lossfunction.

    ### Parameters
    - *ae*: Autoencoder, required
    + The autoencoder to fit.
    - *train_data*: torch.Tensor, required
    + Dataset with the data to train on where samples are in rows
    - *val_data*: torch.Tensor, required
    + Dataset with the data to validate on where samples are in rows
    - *optimizer*: torch.optim.Optimizer, required
    - *batch_size*: int, required
    - *max_epochs*: int, required
    + Training is at most run for this number of epochs
    - *early_stopping*: bool, required
    + Whether to use early stopping or not
    - *check_every*: int, optional, default=1500
    + Scores are calculated every *check_every* iteration.
    - *patience*: int, optional, default=10
    + If the validation score has not increased after *patience* checks
     and early stopping is True, training is stopped.
 
    ### Returns
    - *time*: Time taken for fitting
    """

    start_time = time()
    num_nets = stacked_ae.num_nets

    train_loss = StackedMSELoss(size_average=None, reduce=None,
                                reduction='elementwise_mean')
    val_loss = StackedMSELoss(size_average=None, reduce=None,
                              reduction='stacked_elementwise_mean')

    train_dataset = StackedTensorDataset(train_data, train_data, shuffle=True)
    train_loader = StackedDataLoader(train_dataset,
                                     batch_size=batch_size, shuffle=True)

    val_dataset = StackedTensorDataset(val_data, val_data, shuffle=False)
    val_loader = StackedDataLoader(val_dataset, batch_size=4*batch_size,
                                   shuffle=False)

    logger = logging.getLogger('stacked_ae_fit')

    trainer = create_supervised_trainer(stacked_ae, optimizer, train_loss)

    run_av = metrics.RunningAverage(output_transform=lambda x:
                                    x/len(stacked_ae.net_list),
                                    alpha=0.9)
    run_av.attach(trainer, 'av_loss')
    pbar_train = ProgressBar()
    pbar_train.attach(trainer, ['av_loss'])

    val_loss_metric = StackedLossMetric(val_loss)
    evaluator = create_supervised_evaluator(stacked_ae,
                                            metrics={'stacked_loss':
                                                     val_loss_metric})

    def score_function(evaluator):
        return -evaluator.state.metrics['stacked_loss']

    evaluator.register_events(*EarlyStoppingEvents)
    evaluator.add_event_handler(Events.COMPLETED,
                                StackedEarlyStopping(patience,
                                                     score_function,
                                                     stacked_ae.num_nets))

    evaluator.num_converged = 0
    @evaluator.on(EarlyStoppingEvents.NET_CONVERGED)
    def update_count(evaluator):
        evaluator.num_converged += 1

    if early_stopping:
        evaluator.best_state = {}
        for name, tensor in stacked_ae.state_dict().items():
            # Create a new instance of the tensor on the cpu
            evaluator.best_state[name] = torch.tensor(tensor, device='cpu')

        @evaluator.on(EarlyStoppingEvents.HAS_IMPROVED)
        def store_states(evaluator):
            for name, tensor in stacked_ae.state_dict().items():
                # Create a new instance of the tensor on the cpu
                evaluator.best_state[name] = torch.tensor(tensor, device='cpu')

        @evaluator.on(EarlyStoppingEvents.EARLY_STOPPING_DONE)
        def terminate(evaluator):
            trainer.terminate()

    class CustomEvents(Enum):
        N_ITERATIONS_COMPLETED = "n_iterations_completed"

    trainer.register_events(CustomEvents['N_ITERATIONS_COMPLETED'])
    @trainer.on(Events.ITERATION_COMPLETED, check_every)
    def every_n(trainer, check_every):
        if trainer.state.iteration % check_every == 0:
            trainer.fire_event(CustomEvents.N_ITERATIONS_COMPLETED)

    @trainer.on(CustomEvents.N_ITERATIONS_COMPLETED)
    def run_validation(trainer):
        iteration = trainer.state.iteration
        epoch = trainer.state.epoch

        evaluator.run(val_loader, max_epochs=1)

        msg = (("Iteration {} (Epoch {})" +
                "training error:     {:.5f}, " +
                "mean validation error:   {:.5f}, " +
                "{}/{} nets convered.")
               .format(iteration, epoch,
                       trainer.state.metrics['av_loss'],
                       evaluator.state.metrics['stacked_loss'].mean(),
                       evaluator.num_converged, num_nets))

        logger.info(msg)

        # clear some memory
        evaluator.state.output = None

        if writer is not None:
            writer.add_scalar('train_error_estimate',
                              trainer.state.output, iteration)
        # Write to tensorboard
        for i, loss in enumerate(evaluator.state.metrics['stacked_loss']):
            if writer is not None:
                writer.add_scalar('val_error_{}'.format(i), loss, iteration)

    trainer.run(train_loader, max_epochs=max_epochs)

    if early_stopping:
        # Load the best states
        stacked_ae.load_state_dict(evaluator.best_state)
        stacked_ae.update_list()

    train_time = time()-start_time

    if early_stopping and evaluator.num_converged < num_nets:
        msg = ("Only {}/{} nets converged"
               .format(evaluator.num_converged, num_nets))
        logger.warning(msg)

    msg = ('Trained for {} epochs, {}/{} nets converged.'
           '\n Training took {}s'.format(trainer.state.epoch,
                                         evaluator.num_converged, num_nets,
                                         train_time))
    logger.info(msg)
    return (train_time)
