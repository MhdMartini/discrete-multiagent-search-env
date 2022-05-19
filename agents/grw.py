"""
Guided Random Walk agents class.  
"""

import numpy as np


class GRW:
    def __init__(self,
                 n_actions: int,  # number of actions
                 fov: int,  # field of view as L1 distance from the agent
                 ):
        self.n_actions = n_actions
        self.fov = fov

    def learn(self, s, a, r, sp):
        return

    def choose_action(self, s: np.array):
        raise NotImplemented
