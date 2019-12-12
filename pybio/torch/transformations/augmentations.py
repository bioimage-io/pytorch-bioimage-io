import numpy as np
from .base import SynchronizedTransformation


# TODO do we implement a version for torch and numpy tensors?
# TODO nd ?
# TODO random seed
class RandomAxisRotation(SynchronizedTransformation):
    def __init__(self, apply_to):
        super().__init__(apply_to)
        self.k = 0

    def set_next_state(self):
        self.k = np.random.randint(0, 4)

    # FIXME only works for 2d inputs
    def apply_transformation(self, tensor):
        return np.rot90(tensor, k=self.k)
