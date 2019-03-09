"""
All the stacked datahandling goes here.
"""
import torch
import torch.utils.data as D
import torch.utils.data.dataloader as DL


def shuffle_stacked(stacked_tensor):
    """
    Shuffles each of the layers in a stacked tensor.
    """
    for tensor in stacked_tensor:
        rand_map = torch.randperm(tensor.size(0))
        tensor[:] = tensor[rand_map]


class StackedTensorDataset(D.Dataset):
    """Dataset wrapping tensors.
    The samples are then drawn from the second dimension.
    ### Parameters
    - *tensors* (Tensor): tensors that have
    the same size of the first dimension.
    - *shuffle* (bool): If True shuffles each network in each tensor
"""

    def __init__(self, *tensors, shuffle=True):
        # Ensures all tensors are to equally many nets
        num_nets = tensors[0].size(0)
        assert all(num_nets == tensor.size(0) for tensor in tensors)
        num_samples = tensors[0].size(1)
        # Ensures all tensors have equally many samples
        assert all(num_samples == tensor.size(1) for tensor in tensors)
        self.tensors = tuple(tensor for tensor in tensors)
        if shuffle:
            for tensor in self.tensors:
                shuffle_stacked(tensor)

    def __getitem__(self, index):

        return tuple(tensor[:, index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(1)


class StackedDataLoader(D.DataLoader):
    """Wrapper around D.DataLoader to get the batches in the right format."""

    def __iter__(self):
        return _StackedDataLoaderIter(self)


class _StackedDataLoaderIter(DL._DataLoaderIter):
    def __next__(self):
        batch = super().__next__()
        batch = [s.transpose(0, 1) for s in batch]
        return batch
