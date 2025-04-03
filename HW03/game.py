import matplotlib.pyplot as plt
import numpy as np

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

        self.maze = []
        self.MAZE_HEIGHT = 0
        self.MAZE_WIDTH = 0

        self.RGB_GREY = (.5, .5, .5)
        self.RGB_GREEN = (.5, 1, 0)
        self.RGB_RED = (1, 0, 0)
        self.RGB_YELLOW = (1, 1, 0)
        self.RGB_BLACK = (0, 0, 0)

        self.init_maze()

    def init_maze(self):
        self.maze = [[1, 1, 1, 1, 1, 1, 1, 1, 3],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 2, 1, 1, 1, 1, 1]]

        self.MAZE_HEIGHT = len(self.maze)
        self.MAZE_WIDTH = len(self.maze[0])


    def change_maze(self):
        self.maze = [[1, 1, 1, 1, 1, 1, 1, 1, 3],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 0, 0, 0, 0, 1],
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
        maze_rgb = self.maze.copy()
        maze_rgb = [[self.RGB_GREEN if s == self.OBSTACLE_STATE else self.RGB_GREY if s == self.OK_STATE else
                    self.RGB_YELLOW if s == self.START_STATE else self.RGB_RED for s in row] for row in maze_rgb]
        if state is not None:
            x, y = state
            maze_rgb[x][y] = self.RGB_BLACK
        im = plt.imshow(maze_rgb, origin='lower', interpolation='none', animated=True)
        plt.gca().invert_yaxis()
        return im