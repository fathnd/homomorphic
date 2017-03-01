import torch


class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples


class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(torch.randperm(self.num_samples).long())

    def __len__(self):
        return self.num_samples


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, always in the same order.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.num_samples = len(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.num_samples = len(indices)
        self.indices = indices

    def __iter__(self):
        return iter([self.indices[i] for i in torch.randperm(self.num_samples).long()])

    def __len__(self):
        return self.num_samples
