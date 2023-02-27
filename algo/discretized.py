import gym
import numpy as np


class DiscretizedWrapper:
    #This wrapper converts a Box spaces into a single integer.

    def __init__(self, n_bins=10, low=None, high=None):
        #assert isinstance(env.observation_space, gym.spaces.Box)

        low = low
        high = high

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        #self.discrete_space = gym.spaces.Discrete(n_bins ** low.flatten().shape[0])
        self.discrete_nums = (n_bins) ** (low.flatten().shape[0]+1)

    def _convert_to_one_number(self, digits):
        return sum([d * (self.n_bins ** i) for i, d in enumerate(digits)])
        #return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def discrete(self, continous):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(continous.flatten(), self.val_bins)]
        #print(digits)
        return int(self._convert_to_one_number(digits))

