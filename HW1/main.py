import numpy as np
import matplotlib.pyplot as plt


class NonstationaryBandit:
    def __init__ (self, arms=10, std_dev_q=0.01, std_dev_reward=1):
        self.arms = arms
        self.std_dev_q = std_dev_q
        self.std_dev_reward = std_dev_reward
        self.q_true = np.zeros(arms)

    def step(self):
        self.q_true += np.random.normal(0, self.std_dev_q, self.arms)

    def reward(self, action):
        return np.random.normal(self.q_true[action], self.std_dev_reward)


class Agent:
    def __init__ (self, arms=10, epsilon=0.1, alpha=None):
        self.arms = arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_est = np.zeros(arms)
        self.action_count = np.zeros(arms)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.arms)
        else:
            max_val = np.max(self.q_est)
            max_indices = np.where(self.q_est == max_val)[0]
            return np.random.choice(max_indices)

    def update_q_est(self, action, reward):
        self.action_count[action] += 1
        if self.alpha:
            self.q_est[action] += self.alpha * (reward - self.q_est[action])
        else:
            self.q_est[action] += (1 / self.action_count[action]) * (reward - self.q_est[action])


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    means = np.convolve(values, weights, 'valid')
    stds = np.std(rolling_window(values, window), 1)
    return means, stds


def run_experiment(bandit, agent, steps=10000):
    rewards = np.zeros(steps)
    optimal_action_count = np.zeros(steps)

    for t in range(steps):
        action = agent.select_action()
        reward = bandit.reward(action)
        agent.update_q_est(action, reward)
        bandit.step()

        rewards[t] = reward
        optimal_action_count[t] = (action == np.argmax(bandit.q_true))

    return rewards, optimal_action_count

# experiment setting

arms = 10
steps = 10000
runs = 2000
epsilon = 0.1
alpha = 0.1
rolling_window_size = 25

rewards_avg = np.zeros((2, steps))
optimal_action_count_avg = np.zeros((2, steps))

for run in range(runs):
    bandit = NonstationaryBandit()

    agent_sample_avg = Agent(arms, epsilon, alpha = None)
    agent_constant_step = Agent(arms, epsilon, alpha = alpha)

    rewards1, optimal1 = run_experiment(bandit, agent_sample_avg, steps)
    rewards2, optimal2 = run_experiment(bandit, agent_constant_step, steps)

    rewards_avg[0] += rewards1
    rewards_avg[1] += rewards2
    optimal_action_count_avg[0] += optimal1
    optimal_action_count_avg[1] += optimal2

rewards_avg /= runs
optimal_action_count_avg /= runs

# analysis

fig1, axes1 = plt.subplots(2, 1, figsize = (10, 10))
labels = ['Sample Average', 'Constant Step-Size (α=0.1)']
colors = ['r', 'g']

for i in range(2):
    axes1[0].plot(rewards_avg[i], label = labels[i], color = colors[i])
    axes1[1].plot(optimal_action_count_avg[i] * 100, label = labels[i], color = colors[i])

axes1[0].set_ylabel('Average Reward')
axes1[1].set_ylabel('% Optimal Action')
axes1[1].set_xlabel('Steps')
axes1[0].set_xlim(0, steps)
axes1[1].set_xlim(0, steps)
axes1[0].legend()
axes1[1].legend()
axes1[0].grid(True)
axes1[1].grid(True)


fig2, axes2 = plt.subplots(2, 1, figsize = (10, 10))
labels = ['Sample Average', 'Constant Step-Size (α=0.1)']
colors = ['r', 'g']

for i in range(2):
    x = range(steps)

    y1, stds1 = moving_average(rewards_avg[i], window=rolling_window_size)
    y2, stds2 = moving_average(optimal_action_count_avg[i], window=rolling_window_size)
    x1 = x[len(x) - len(y1):]
    x2 = x[len(x) - len(y2):]

    axes2[0].plot(x1, y1, label = labels[i], color = colors[i])
    axes2[0].fill_between(x1, y1 - stds1, y1 + stds1, alpha=0.2, color = colors[i])
    axes2[1].plot(x2, y2, label = labels[i], color = colors[i])
    axes2[1].fill_between(x2, y2 - stds2, y2 + stds2, alpha=0.2, color = colors[i])

axes2[0].set_ylabel('Average Reward')
axes2[1].set_ylabel('% Optimal Action')
axes2[1].set_xlabel('Steps')
axes2[0].set_xlim(0, steps)
axes2[1].set_xlim(0, steps)
axes2[0].legend()
axes2[1].legend()
axes2[0].grid(True)
axes2[1].grid(True)
plt.show()



