# TODO would be nice to have something similar to inferno's batch_function etc.
# functionality, but more flexible w.r.t implementing different dimensions.
from typing import List, Optional, Sequence, Tuple

import torch

import pybio.transformations

class Transformation(pybio.transformations.Transformation):
    pass

class IndependentTransformation(Transformation):
    """ Transformation that can be applied to all input tensors independently.
    """
    pass


class SynchronizedTransformation(Transformation):
    """ Transformation for which application to all tensors is synchronized.
    This means, some state must be known before applying it to the tensors,
    e.g. the degree before a random rotation
    """
    def apply(self, *tensors):
        # TODO the state might depend on some tensor properties (esp. shape)
        # inferno solves this with the 'set_random_state' and 'get_random_state' construction
        # here, we could just pass *tensors to set_next_state
        self.set_next_state()
        return super().apply(*tensors)


class Loss(pybio.transformations.Loss):
    pass
