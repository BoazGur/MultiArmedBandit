import matplotlib.pyplot as plt
import numpy as np

from AdvertisementBandit import AdvertisementBandit
from Solvers import Solver, EpsilonGreedy, UCB1, BayesianUCB, ThompsonSampling


def plot_results(solvers, solver_names, figname):
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure(figsize=(14, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2: Probabilities estimated by solvers.
    # sort probabilities by magnitude
    # plot probabilities by magnitude
    sorted_indices = sorted(range(b.num_bandits), key=lambda x: b.probs[x])
    ax2.plot(range(b.num_bandits), [b.probs[x]
             for x in sorted_indices], 'k--', markersize=12)

    # for each solver plot the estimated probabilties
    for s in solvers:
        ax2.plot(range(b.num_bandits), [s.estimated_probs[x]
                 for x in sorted_indices], 'x', markeredgewidth=2)
    ax2.set_xlabel('Actions sorted by their true probabilities ' + r'$\theta$')
    ax2.set_ylabel('Estimated probabilities')
    ax2.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 3: Action counts
    for s in solvers:
        ax3.plot(range(b.num_bandits), np.array(s.counts) /
                 float(len(solvers[0].regrets)), ls='-', lw=2)
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Frac. # trials')
    ax3.grid('k', ls='--', alpha=0.3)

    plt.savefig(figname)


def experiment(num_bandits, steps):
    bandit = AdvertisementBandit(num_bandits)
    print("Randomly generated Bernoulli bandit has reward probabilities:\n")
    for element in bandit.probs:
        print(f"{element:.4f}")
    print("The best machine has index: {} and proba: {:.6f}".format(
        max(range(num_bandits), key=lambda i: bandit.probs[i]), max(bandit.probs)))

    # create solvers
    test_solvers = [
        # EpsilonGreedy(b, 0),
        # EpsilonGreedy(b, 1),
        EpsilonGreedy(bandit, eps=0.01),
        UCB1(bandit),
        BayesianUCB(bandit, 3, 1, 1),
        ThompsonSampling(bandit, 1, 1)
    ]
    names = [
        # 'Full-exploitation',
        # 'Full-exploration',
        r'$\epsilon$' + '-Greedy',
        'UCB1',
        'Bayesian UCB',
        'Thompson Sampling'
    ]

    # loop on solvers and solve problem
    for solver in test_solvers:
        print('Running ' + solver.name)
        solver.run(steps)

    plot_results(test_solvers, names,
                 "results_Advertisement_K{}_N{}.png".format(num_bandits, steps))


def main():
    experiment(num_bandits=10, steps=5000)


if __name__ == '__main__':
    main()
