import numpy as np


class RNG:
    def __init__(self, seed=None):
        self.numpy_rng = np.random.default_rng(seed)

    def random_float(self, min_value=None, max_value=None, size=None):
        if min_value is None and max_value is None:
            return self.numpy_rng.random(size)  # Returns a random float between 0 and 1
        return min_value + self.numpy_rng.random(size) * (max_value - min_value)

    def rand_np_int(self, min_value=0, max_value=None, size=None):
        return self.numpy_rng.integers(low=min_value, high=max_value, size=size, endpoint=True)
