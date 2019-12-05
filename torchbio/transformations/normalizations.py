from .base import IndependentTransformation


class NormalizeZeroMeanUnitVariance(IndependentTransformation):
    """ Sigmoid activation
    """

    def __init__(self, eps=1.0e-6, mean=None, std=None, **super_kwargs):
        super().__init__(**super_kwargs)
        self.eps = eps
        self.mean = mean
        self.std = std

    def apply_to_one(self, tensor):
        mean = tensor.mean() if self.mean is None else self.mean
        std = tensor.std() if self.std is None else self.std
        return (tensor - mean) / (std + self.eps)
