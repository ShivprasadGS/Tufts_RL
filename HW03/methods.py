import os
import numpy as np
import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt


class TabularDynaQ:
    def __init__(self, maze, params):
        self.maze = maze
        self.params = params

        self.Q = np.zeros((self.maze.MAZE_HEIGHT, self.maze.MAZE_WIDTH, len(self.maze.actions)))

        self.model = np.empty((self.maze.MAZE_HEIGHT, self.maze.MAZE_WIDTH, len(self.maze.actions)), dtype=list)

        self.time = 0
        self.times = np.zeros((self.maze.MAZE_HEIGHT, self.maze.MAZE_WIDTH, len(self.maze.actions)))

    def choose_action(self, state, method, deterministic=False):
        if method == 'Dyna-Q' or method == 'Dyna-Q+':
            if np.random.binomial(1, self.params.epsilon) == 1 and not deterministic:
                return random.choice(self.maze.actions)
            else:
                values = self.Q[state[0], state[1], :]
                return random.choice([action for action, value in enumerate(values) if value == np.max(values)])

        elif method == 'Modified-Dyna-Q+':
            if np.random.binomial(1, self.params.epsilon) == 1 and not deterministic:
                return random.choice(self.maze.actions)
            elif not deterministic:
                values = self.Q[state[0], state[1], :] + \
                         self.params.k * np.sqrt(self.time - self.times[state[0], state[1], :])
                return random.choice([action for action, value in enumerate(values) if value == np.max(values)])
            else:
                values = self.Q[state[0], state[1], :]
                return random.choice([action for action, value in enumerate(values) if value == np.max(values)])

    def resolve_maze(self, method):
        steps = 0
        reward_ = 0
        rewards = np.zeros(self.params.steps)
        self.maze.init_maze()

        has_changed = False

        while steps < self.params.steps:
            state = random.choice(self.maze.get_state_locations(self.maze.START_STATE))
            prev_steps = steps

            while state not in self.maze.get_state_locations(self.maze.GOAL_STATE):
                steps += 1

                action = self.choose_action(state, method)
                new_state, reward = self.maze.take_action(state, action)

                self.Q[state[0], state[1], action] += self.params.alpha * (reward + self.params.gamma * np.max(self.Q[new_state[0], new_state[1], :])
                                         - self.Q[state[0], state[1], action])

                self.model[state[0], state[1], action] = [new_state, reward]

                self.time += 1
                self.times[state[0], state[1], action] = self.time

                for _ in range(self.params.n):
                    p_state = random.choice(
                        [[i, j] for i in np.arange(self.maze.MAZE_HEIGHT) for j in np.arange(self.maze.MAZE_WIDTH)
                         if not all(v is None for v in self.model[i, j, :])])

                    if method == 'Dyna-Q' or method == 'Modified-Dyna-Q+':
                        p_action = random.choice(
                            [a for a in self.maze.actions if self.model[p_state[0], p_state[1], a] is not None])

                        p_new_state, p_reward = self.model[p_state[0], p_state[1], p_action]

                        self.Q[p_state[0], p_state[1], p_action] += self.params.alpha * (
                            p_reward + self.params.gamma * np.max(self.Q[p_new_state[0], p_new_state[1], :])
                            - self.Q[p_state[0], p_state[1], p_action])

                    elif method == 'Dyna-Q+':
                        p_action = random.choice(self.maze.actions)

                        if self.model[p_state[0], p_state[1], p_action] is not None:
                            p_new_state, p_reward = self.model[p_state[0], p_state[1], p_action]

                        else:
                            p_new_state, p_reward = p_state, 0

                        p_reward += self.params.k * np.sqrt(self.time - self.times[p_state[0], p_state[1], p_action])

                        self.Q[p_state[0], p_state[1], p_action] += self.params.alpha * (
                            p_reward + self.params.gamma * np.max(self.Q[p_new_state[0], p_new_state[1], :])
                            - self.Q[p_state[0], p_state[1], p_action])

                state = new_state

            rewards[prev_steps:steps] = reward_
            reward_ += 1
            if steps > self.params.changing_steps and not has_changed:
                self.maze.change_maze()
                has_changed = True

        return rewards

    def sample_episode(self, method):
        states = [random.choice(self.maze.get_state_locations(self.maze.START_STATE))]
        steps = 0

        while states[steps] not in self.maze.get_state_locations(self.maze.GOAL_STATE) and steps < self.params.max_steps:
            action = self.choose_action(states[steps], method, deterministic=True)

            new_state, _ = self.maze.take_action(states[steps], action)
            states.append(new_state)

            steps += 1

        ims = []

        fig = plt.figure()
        plt.title(method)
        for state in states:
            im = self.maze.print_maze(state)
            ims.append([im])

        anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=0)

        os.makedirs("./results", exist_ok=True)

        if method == 'Dyna-Q':
            anim.save("./results/animation-Dyana-Q.gif", writer="pillow", fps=10)
        elif method == 'Dyna-Q+':
            anim.save("./results/animation-Dyana-Q+.gif", writer="pillow", fps=10)
        elif method == 'Modified-Dyna-Q+':
            anim.save("./results/animation-Modified-Dyna-Q+.gif", writer="pillow", fps=10)
        plt.show()

