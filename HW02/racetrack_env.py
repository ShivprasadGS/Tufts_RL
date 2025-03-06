import numpy as np
from gymnasium import Env
import pygame

STARTING = 0.8
FINISHING = 0.4

# Racetrack Environment
class Racetrack(Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 6}

    def __init__(self, track_map: str, render_mode: str = None, size: int = 20):

        self.size = size

        assert track_map in ['a', 'b']
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        filename = 'track_a.npy' if track_map == 'a' else 'track_b.npy'
        with open('./racetrack_designs/' + filename, 'rb') as f:
            self.track_map = np.load(f)

        # initialize parameters for pygame
        self.window_size = self.track_map.shape
        # pygame's coordinate system - transpose of that of numpy array
        self.window_size = (self.window_size[1] * self.size, self.window_size[0] * self.size)
        self.window = None
        self.clock = None
        self.truncated = False

        # get the starting states
        self.start_states = np.dstack(np.where(self.track_map == STARTING))[0]

        self.nS = (*self.track_map.shape, 5, 9)         # observation space
        self.nA = 9                                     # action space
        self.state = None                               # initialize state
        self.speed = None                               # initialize speed

        # Mapping an integer action to acceleration tuple
        self._action_to_acceleration = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, -1),
            7: (1, 0),
            8: (1, 1)
        }

    # get observation
    def _get_obs(self):

        return (*self.state, *self.speed)

    # check if the race car goes crosses the finish line
    def _check_finish(self):

        finish_states = np.where(self.track_map == FINISHING)
        rows = finish_states[0]
        col = finish_states[1][0]

        if self.state[0] in rows and self.state[1]>=col:
            return True

        return False

    # check if the race car goes out of the track
    def _check_out_track(self, next_state):

        row, col = next_state
        H, W = self.track_map.shape

        # check if car goes out of the boundary
        if row<0 or row>=H or col<0 or col>=W:
            return True

        # check if car runs into the gravel
        if self.track_map[next_state[0], next_state[1]] == 0:
            return True

        # check if part of the path runs into the gravel
        for row_step in range(self.state[0], row, -1):
            if self.track_map[row_step, self.state[1]] == 0:
                return True
        for col_step in range(self.state[1], col, 1 if col > self.state[1] else -1):
            if self.track_map[row, col_step] == 0:
                return True

        return False

    # reset the car to one of the starting positions
    def reset(self):

        start_idx = np.random.choice(self.start_states.shape[0])
        self.state = self.start_states[start_idx]
        self.speed = (0, 0)

        if self.render_mode == 'human':
            self.render(self.render_mode)

        return self._get_obs()

    # what needs to be done in each step
    def step(self, action):

        # get new acceleration and update the position
        new_state = np.copy(self.state)
        y_act, x_act = self._action_to_acceleration[action]

        temp_y_acc = self.speed[0] + y_act
        temp_x_acc = self.speed[1] + x_act

        if temp_y_acc < -4:
            temp_y_acc = -4
        if temp_y_acc > 0:
            temp_y_acc = 0
        if temp_x_acc < -4:
            temp_x_acc = -4
        if temp_x_acc > 4:
            temp_x_acc = 4

        new_state[0] += temp_y_acc
        new_state[1] += temp_x_acc

        terminated = False

        # check if the next position crosses the finish line
        if self._check_finish():
            terminated = True

        # check if the next position is out of track
        elif self._check_out_track(new_state):
            self.reset()

        else:
            self.state = new_state
            self.speed = (temp_y_acc, temp_x_acc)

        if self.render_mode == 'human':
            self.render(self.render_mode)

        return self._get_obs(), -1, terminated, self.truncated

    # visualize race map
    def render(self, mode):

        if self.window is None:
            pygame.init()
            pygame.display.set_caption('Racetrack')
            if mode == 'human':
                self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        rows, cols = self.track_map.shape
        self.window.fill((255, 255, 255))

        # Draw the map
        for row in range(rows):
            for col in range(cols):
                cell_val = self.track_map[row, col]
                # Draw finishing cells
                if cell_val == FINISHING:
                    fill = (235, 52, 52)
                    pygame.draw.rect(self.window, fill, (col * self.size, row * self.size, self.size, self.size), 0)
                # Draw starting cells
                elif cell_val == STARTING:
                    fill = (61, 227, 144)
                    pygame.draw.rect(self.window, fill, (col * self.size, row * self.size, self.size, self.size), 0)

                color = (120, 120, 120)
                # Draw gravels
                if cell_val == 0:
                    color = (255, 255, 255)
                # Draw race track
                elif cell_val == 1:
                    color = (160, 160, 160)

                pygame.draw.rect(self.window, color, (col * self.size, row * self.size, self.size, self.size), 1)

        # Draw the car
        pygame.draw.rect(self.window, (86, 61, 227),
                         (self.state[1] * self.size, self.state[0] * self.size, self.size, self.size), 0)

        if mode == "human":
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window = None
                    pygame.quit()
                    self.truncated = True
            self.clock.tick(self.metadata['render_fps'])

# run this script when this file is called:
if __name__ == '__main__':

    #render_mode = 'human'
    render_mode = None
    env = Racetrack('a', render_mode=render_mode, size=20)
    env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:

        action = np.random.choice(env.nA)
        observation, reward, terminated, truncated = env.step(action)
        total_reward += reward

        if terminated:
            print(observation, reward, terminated, total_reward)




