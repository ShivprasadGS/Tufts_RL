import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.ndimage import uniform_filter
import seaborn as sns

from racetrack_env import Racetrack

# behavior policy
def behavior_pi(state:tuple,
                nA: int,
                target_pi: np.ndarray,
                epsilon: float) -> tuple:

    rand_val = np.random.rand()
    greedy_act = target_pi[state]

    if rand_val > epsilon:
        return greedy_act, (1 - epsilon + epsilon / nA)
    else:
        action = np.random.choice(nA)
        if action == greedy_act:
            return action, (1 - epsilon + epsilon / nA)
        else:
            return action, epsilon / nA

# plot result
def plot_result(value_hist: dict, total_episodes) -> None:

    # Set Seaborn style for better aesthetics
    sns.set_style("whitegrid")

    # Define line width and font properties
    line_width = 2.0
    fontdict = {'fontsize': 14, 'fontweight': 'bold'}

    # Create figure
    plt.figure(figsize=(12, 7), dpi=200)
    plt.ylim((-500.0, 0.0))
    plt.grid(color='gray', linestyle='dashed', alpha=0.4)  # Dashed grid for a modern look
    plt.margins(0.02)

    # Customize axis spines
    ax = plt.gca()
    for pos, spine in ax.spines.items():
        if pos in ["top", "right"]:
            spine.set_visible(False)  # Hide top and right spines
        else:
            spine.set_linewidth(1.8)

    # Generate x values
    x = np.arange(total_episodes)
    plt.xscale('log')
    plt.xticks(
        [1, 10, 100, 1000, 10_000, 100_000, 1_000_000],
        ['1', '10', '100', '1K', '10K', '100K', '1M']
    )

    # Use a more refined color palette
    colors = sns.color_palette("Set2", n_colors=len(value_hist))

    # Plot the data
    for i, (key, value) in enumerate(value_hist.items()):
        title, label = key.split(',')
        plt.plot(
            x, uniform_filter(value, size=30),  # Slightly smoother curves
            linewidth=line_width,
            label=label,
            color=colors[i],
            alpha=0.9
        )

    # Set labels and title
    plt.title(f"{title} Training Record", fontdict=fontdict, pad=15)
    plt.xlabel("Episodes (Log Scale)", fontdict=fontdict, labelpad=10)
    plt.ylabel("Rewards", fontdict=fontdict, labelpad=10)
    plt.legend(frameon=True, fancybox=True, shadow=True)

    # Save and show the plot
    plt.savefig(f'./plots/{"_".join(title.lower().split())}.png', bbox_inches='tight')
    plt.show()

# off-policy monte carlo algorithm:
def off_policy_monte_carlo(total_episodes: int,
                           track_map:str, render_mode: str,
                           zero_acc: bool = False) -> tuple:

    gamma = 0.9
    epsilon = 0.1

    env = Racetrack(track_map, render_mode, size=20)
    action_space = env.nA
    observation_space = env.nS

    Q = np.random.normal(size=(*observation_space, action_space))
    Q -= 500

    C = np.zeros_like(Q)

    target_pi = np.argmax(Q, axis=-1)

    reward_hist = np.zeros(shape=(total_episodes), dtype=np.float32)

    for i in range(total_episodes):

        trajectory = []
        terminated = False
        state = env.reset()
        (action, act_prob) = behavior_pi(state, env.nA, target_pi, epsilon)

        ttl_reward = 0

        # generate trajectory using behavior policy
        while not terminated:

            if zero_acc and np.random.rand() <= 0.1:
                non_acc_act = 4  # check env._action_to_acceleration
                observation, reward, terminated, _ = env.step(non_acc_act)
            else:
                observation, reward, terminated, _ = env.step(action)

            ttl_reward += reward
            trajectory.append((state, action, reward, act_prob))
            state = observation
            (action, act_prob) = behavior_pi(state, env.nA, target_pi, epsilon)

        G = 0.
        W = 1.

        # loop inversely to update G and Q values:
        while trajectory:

            (state, action, reward, act_prob) = trajectory.pop()
            G = gamma * G + reward
            C[state][action] = C[state][action] + W
            Q[state][action] = Q[state][action] + (W / C[state][action]) * (G - Q[state][action])

            target_pi[state] = np.argmax(Q[state])

            if action != target_pi[state]:
                break

            W = W * (1 / act_prob)

        reward_hist[i] = ttl_reward

        if i % 1000 == 0:
            print(f'Episode: {i}, reward: {ttl_reward}, epsilon: {epsilon}')

    return reward_hist, Q

# run this script when this file is run
if __name__ == '__main__':

    train = True                # decides if we are in training phase
    track_sel = 'a'
    total_episodes = 1_000_000

    if train:

        reward_hist_dict = dict()
        Q_dict = dict()

        for i in range(2):
            track_name = f'Track {track_sel.capitalize()}'
            use_zero_acc = 'with zero acceleration' if i else 'without zero acceleration'
            key = track_name + ',' + use_zero_acc

            reward_hist, Q = off_policy_monte_carlo(total_episodes, track_sel, None, i)
            reward_hist_dict[key] = reward_hist
            Q_dict[key] = Q

            plot_result(reward_hist_dict, total_episodes)
            with open(f'./history/track_{track_sel}.pkl', 'wb') as f:
                pickle.dump(Q_dict, f)

    else:  # Evaluate the Q values and plot sample paths

        with open(f'./history/track_{track_sel}.pkl', 'rb') as f:
            Q_dict = pickle.load(f)

        key = list(Q_dict.keys())[0]
        Q = Q_dict[key]
        policy = np.argmax(Q, axis=-1)  # greedy policy

        env = Racetrack(track_sel, None, 20)
        plt.figure(dpi=200)
        plt.suptitle('Sample trajectories', size=12, weight='bold')

        for i in range(6):
            track_map = np.copy(env.track_map)
            state = env.reset()
            terminated = False
            while not terminated:
                track_map[state[0], state[1]] = 0.6
                action = policy[state]
                next_state, reward, terminated,_ = env.step(action)
                state = next_state

            ax = plt.subplot(2, 3, i + 1)
            ax.axis('off')
            ax.imshow(track_map, cmap='GnBu')
            sns.heatmap(track_map, linewidths=1)
        plt.tight_layout()
        plt.savefig(f'./plots/track_{track_sel}_paths.png')
        plt.show()
