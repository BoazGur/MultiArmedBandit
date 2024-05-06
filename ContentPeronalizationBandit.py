from time import time
import numpy as np

from Bandit import Bandit


class ContentPersonalizationBandit(Bandit):
    def __init__(self, num_bandits, probs=None):
        assert probs is None or len(probs) == num_bandits, 'probs should match num_bandits'

        self.num_bandits = num_bandits # Number of categories

        # Set the bandits' rewards randomly
        if probs is None:
            np.random.seed(int(time()))
            self.probs = [np.random.random() for _ in range(self.num_bandits)]
        else:
            self.probs = probs
        
        # Maximal reward
        self.best_prob = max(self.probs)

    def generate_reward(self, index):
        # The player selected the content category of index
        if np.random.random() < self.probs[index]:
            return 1 # User clicks on ad
        
        return 0