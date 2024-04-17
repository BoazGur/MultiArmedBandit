from __future__ import division

from scipy.stats import beta
from time import time
import numpy as np

from ContentPeronalizationBandit import ContentPersonalizationBandit
from AdvertisementBandit import AdvertiseBandit


class Solver(object):
    def __init__(self, bandit):
        assert isinstance(bandit, AdvertiseBandit) or isinstance(
            bandit, ContentPersonalizationBandit)

        np.random.seed(int(time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.num_bandits
        self.actions = []  # Bandit ids
        self.regret = 0  # Cumulative regret
        self.regrets = [0.]  # History of cumulative regret

        self.name = 'General'

    def update_regret(self, index):
        self.regret += self.bandit.best_prob - \
            self.bandit.probs[index]  # "loss"
        self.regrets.append(self.regret)

    @property
    def estimated_probs(self):
        raise NotImplementedError

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None

        for k in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1

            # Store results
            self.actions.append(i)
            self.update_regret(i)


class EpsilonGreedy(Solver):
    def __init__(self, bandit, eps, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)

        self.name = 'EpsilonGreedy'

        assert 0. <= eps <= 1.0

        self.eps = eps
        self.estimates = [init_prob] * self.bandit.num_bandits

    @property
    def estimated_probs(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.eps:
            i = np.random.randint(0, self.bandit.num_bandits)  # explore
        else:
            i = np.argmax(self.estimates)  # exploit

        reward = self.bandit.generate_reward(i)
        self.estimates[i] += (reward - self.estimates[i]) / (self.counts[i] + 1)

        return i


class UCB1(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(UCB1, self).__init__(bandit)

        self.name = 'UCB1'

        self.t = 0
        self.estimates = [init_prob] * self.bandit.num_bandits

    @property
    def estimated_probas(self):
        return self.estimates

    def run_one_step(self):
        self.t += 1

        i = max(range(self.bandit.num_bandits),
                key=lambda x: self.estimates[x] + np.sqrt(2 * np.log(self.t) / (1 + self.counts[x])))

        reward = self.bandit.generate_reward(i)

        self.estimates[i] += (reward - self.estimates[i]) / (self.counts[i] + 1)

        return i


class BayesianUCB(Solver):
    def __init__(self, bandit, c=3, init_a=1, init_b=1, init_prob=1.0):
        super(BayesianUCB, self).__init__(bandit)

        self.name = 'BayesianUCB'

        self.c = c
        self._as = [init_a] * self.bandit.num_bandits
        self._bs = [init_b] * self.bandit.num_bandits

        self.Xestimates = [init_prob] * self.bandit.num_bandits
        self.X2estimates = [init_prob] * self.bandit.num_bandits

    @property
    def estimated_probas(self):
        return [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.num_bandits)]

    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        i = max(range(self.bandit.num_bandits),
                key=lambda x: self._as[x] / float(self._as[x] + self._bs[x]) + beta.std(self._as[x], self._bs[x]) * self.c)

        reward = self.bandit.generate_reward(i)

        # Update Gaussian posterior
        self._as[i] += reward
        self._bs[i] += (1 - reward)

        self.Xestimates[i] += (reward - self.Xestimates[i]) / (self.counts[i] + 1)
        self.X2estimates[i] += (reward**2 -self.X2estimates[i]) / (self.counts[i] + 1)

        return i


class ThompsonSampling(Solver):
    def __init__(self, bandit, init_a=1, init_b=1):
        super(ThompsonSampling, self).__init__(bandit)

        self.name = 'ThompsonSampling'

        self._as = [init_a] * self.bandit.num_bandits
        self._bs = [init_b] * self.bandit.num_bandits

    @property
    def estimated_probas(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.num_bandits)]

    def run_one_step(self):
        samples = [np.random.beta(self._as[x], self._bs[x])
                   for x in range(self.bandit.num_bandits)]
        
        i = max(range(self.bandit.num_bandits), key=lambda x: samples[x])
        reward = self.bandit.generate_reward(i)

        self._as[i] += reward
        self._bs[i] += (1 - reward)

        return i
