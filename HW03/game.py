import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Maze:
    def __init__(self):
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        self.OBSTACLE_STATE = 0
        self.OK_STATE = 1
        self.START_STATE = 2
        self.GOAL_STATE = 3

        pastel_palette = sns.color_palette("pastel", 10)
        self.RGB_OBSTACLE = pastel_palette[2]
        self.RGB_OK = pastel_palette[7]
        self.RGB_START = pastel_palette[4]
        self.RGB_GOAL = pastel_palette[3]
        self.RGB_AGENT = (0.1, 0.1, 0.1)

        self.state_to_color = {
            self.OBSTACLE_STATE: self.RGB_OBSTACLE,
            self.OK_STATE:       self.RGB_OK,
            self.START_STATE:    self.RGB_START,
            self.GOAL_STATE:     self.RGB_GOAL
        }

        self.maze = []
        self.MAZE_HEIGHT = 0
        self.MAZE_WIDTH = 0

        self.init_maze()

    def init_maze(self):
        self.maze = [[1, 1, 1, 1, 1, 1, 1, 1, 3],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 2, 1, 1, 1, 1, 1]]

        self.MAZE_HEIGHT = len(self.maze)
        self.MAZE_WIDTH = len(self.maze[0])


    def change_maze(self):
        self.maze = [[1, 1, 1, 1, 1, 1, 1, 1, 3],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 2, 1, 1, 1, 1, 1]]

        self.MAZE_HEIGHT = len(self.maze)
        self.MAZE_WIDTH = len(self.maze[0])


    def take_action(self, state, action):
        x, y = state

        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.MAZE_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.MAZE_WIDTH - 1)

        if self.maze[x][y] == self.OBSTACLE_STATE:
            # Stay in the original state if it's an obstacle
            x, y = state

        if self.maze[x][y] == self.GOAL_STATE:
            reward = 1.0
        else:
            reward = 0.0

        return [x, y], reward

    def get_state_locations(self, state_type):
        return [[i, j] for i in np.arange(self.MAZE_HEIGHT) for j in np.arange(self.MAZE_WIDTH) if
                self.maze[i][j] == state_type]

    def print_maze(self, state=None):
        maze_rgb = [[self.state_to_color.get(s, self.RGB_OK)
                     for s in row]
                    for row in self.maze]

        if state is not None:
            x, y = state
            if 0 <= x < self.MAZE_HEIGHT and 0 <= y < self.MAZE_WIDTH:
                maze_rgb[x][y] = self.RGB_AGENT

        ax = plt.gca()
        im = ax.imshow(maze_rgb, origin='upper', interpolation='none', animated=True)
        ax.set_xticks(np.arange(-.5, self.MAZE_WIDTH, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.MAZE_HEIGHT, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=1) # White grid lines
        ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)

        return im
