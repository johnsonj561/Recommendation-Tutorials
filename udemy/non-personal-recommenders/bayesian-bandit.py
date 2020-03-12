# instead of playing slot machine one at a time
# on each round we rank each bandit according to samples from their posterior
# model each slot machine as a separate Beta distribution, i.e. model each machine's win rate
# draw a sample from each Beta distribution and sort them in order, i.e. the ranking
# pick top item from ranked list, and play that slot machine
# this is analagous to (CTR) clicks/impressions on a website
# Bayesian approach lets us treat click through rate measurements as random variables,
# and explore-exploit is handled by random variables

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit(object):
  def __init__(self, p):
    self.p = p
    self.a = 1
    self.b = 1

  def pull(self):
    return np.random.random() < self.p

  def sample(self):
    return np.random.beta(self.a, self.b)

  def update(self, x):
    self.a += x
    self.b += 1 - x


def plot(bandits, trial):
  x = np.linspace(0, 1, 200)
  for b in bandits:
    y = beta.pdf(x, b.a, b.b)
    plt.plot(x, y, label='real p: %.4f' % b.p)
  plt.title('Bandit distributions after %s trials' % trial)
  plt.legend()
  plt.show()

def experiment():
  bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

  sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]

  for i in range(NUM_TRIALS):
    bestb = None
    maxsample = -1
    allsamples = []
    for b in bandits:
      sample = b.sample()
      allsamples.append('%.4f' % sample)
      if sample > maxsample:
        maxsample = sample
        bestb = b
    if i in sample_points:
      print('current samples: %s' % allsamples)
      plot(bandits, i)

    x = bestb.pull()
    bestb.update(x)

if __name__ == '__main__':
  experiment()
