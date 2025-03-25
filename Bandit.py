# """
#   Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
# """
# ############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib as plt
plt.use('Agg') 
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)


class Bandit(ABC):
    """ """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        """ """
        pass

    @abstractmethod
    def update(self, chosen_arm, reward):
        """

        Args:
          chosen_arm: 
          reward: 

        Returns:

        """
        pass

    @abstractmethod
    def experiment(self):
        """ """
        pass

    @abstractmethod
    def report(self):
        """ """
        pass


# #--------------------------------------#

class Visualization():
    """ """

    def plot1(self, rewards_eg, rewards_ts):
        """

        Args:
          rewards_eg: 
          rewards_ts: 

        Returns:

        """
        plt.figure()
        plt.plot(rewards_eg, label='Epsilon-Greedy')
        plt.plot(rewards_ts, label='Thompson Sampling')
        plt.xlabel('Trial')
        plt.ylabel('Reward')
        plt.title('Reward over Time')
        plt.legend()
        plt.grid()
        plt.savefig('plot1.png') # I saved the plot here.
        # plt.show() # apply this on your pc, please, my python does not support this function.

    def plot2(self, cum_eg, cum_ts):
        """

        Args:
          cum_eg: 
          cum_ts: 

        Returns:

        """
        plt.figure()
        plt.plot(cum_eg, label='Epsilon-Greedy')
        plt.plot(cum_ts, label='Thompson Sampling')
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards')
        plt.legend()
        plt.grid()
        plt.savefig('plot2.png') # I saved the plot here.
        # plt.show() # apply this on your pc, please, my python does not support this function.


# #--------------------------------------#

class EpsilonGreedy(Bandit):
    """ """
    def __init__(self, p):
        self.p = p
        self.n = len(p)
        self.counts = np.zeros(self.n)
        self.values = np.zeros(self.n)
        self.rewards = []
        self.cumulative_rewards = []
        self.trials = 20000
        self.data = []

    def __repr__(self):
        return f"EpsilonGreedy(p={self.p})"

    def pull(self):
        """ """
        t = len(self.rewards) + 1
        epsilon = 1 / t
        if np.random.rand() < epsilon:
            return np.random.randint(self.n)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        """

        Args:
          chosen_arm: 
          reward: 

        Returns:

        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = (1 - 1/n) * value + (1/n) * reward

    def experiment(self):
        """ """
        for t in range(self.trials):
            arm = self.pull()
            reward = np.random.normal(self.p[arm], 1)
            self.update(arm, reward)
            self.rewards.append(reward)
            self.cumulative_rewards.append(np.sum(self.rewards))
            self.data.append([arm, reward, 'EpsilonGreedy'])

    def report(self):
        """ """
        df = pd.DataFrame(self.data, columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv('rewards.csv', mode='a', header=False, index=False)
        avg_reward = np.mean(self.rewards)
        regret = np.sum(np.max(self.p) - np.array(self.rewards))
        logger.info(f"[EpsilonGreedy] Average reward: {avg_reward:.4f}")
        logger.info(f"[EpsilonGreedy] Cumulative regret: {regret:.4f}")


# #--------------------------------------#

class ThompsonSampling(Bandit):
    """ """
    def __init__(self, p):
        self.p = p
        self.n = len(p)
        self.trials = 20000
        self.rewards = []
        self.cumulative_rewards = []
        self.data = []
        self.mu = np.zeros(self.n)
        self.lambda_ = np.ones(self.n)

    def __repr__(self):
        return f"ThompsonSampling(p={self.p})"

    def pull(self):
        """ """
        sampled = np.random.normal(self.mu, 1/np.sqrt(self.lambda_))
        return np.argmax(sampled)

    def update(self, chosen_arm, reward):
        """

        Args:
          chosen_arm: 
          reward: 

        Returns:

        """
        self.lambda_[chosen_arm] += 1
        self.mu[chosen_arm] = (self.lambda_[chosen_arm] * self.mu[chosen_arm] + reward) / (self.lambda_[chosen_arm] + 1)

    def experiment(self):
        """ """
        for t in range(self.trials):
            arm = self.pull()
            reward = np.random.normal(self.p[arm], 1)
            self.update(arm, reward)
            self.rewards.append(reward)
            self.cumulative_rewards.append(np.sum(self.rewards))
            self.data.append([arm, reward, 'ThompsonSampling'])

    def report(self):
        """ """
        df = pd.DataFrame(self.data, columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv('rewards.csv', mode='a', header=False, index=False)
        avg_reward = np.mean(self.rewards)
        regret = np.sum(np.max(self.p) - np.array(self.rewards))
        logger.info(f"[ThompsonSampling] Average reward: {avg_reward:.4f}")
        logger.info(f"[ThompsonSampling] Cumulative regret: {regret:.4f}")


def comparison():
    """ """
    # think of a way to compare the performances of the two algorithms VISUALLY and 
    Bandit_Reward = [1, 2, 3, 4]
    eg = EpsilonGreedy(Bandit_Reward)
    ts = ThompsonSampling(Bandit_Reward)

    eg.experiment()
    ts.experiment()

    eg.report()
    ts.report()

    viz = Visualization()
    viz.plot1(eg.rewards, ts.rewards)
    viz.plot2(eg.cumulative_rewards, ts.cumulative_rewards)


if __name__=='__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    comparison()
