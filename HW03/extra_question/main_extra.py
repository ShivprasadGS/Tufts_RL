import matplotlib.pyplot as plt
import numpy as np

from game_extra import Maze
from params_extra import Params
from methods_extra import TabularDynaQ


def show_results(results):
    for result in results:
        rewards, method = result
        plt.plot(rewards, label=method)
    plt.legend(loc='upper left')
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative reward')
    plt.title('Exercise 8.4: Blocking Maze')
    plt.grid(True)
    plt.savefig('./results/results.png')



def gridworld_experiment():
    maze = Maze()
    params = Params()

    results = []
    for method in params.methods:
        rewards = np.zeros(params.steps)

        for run in range(params.runs):
            print('Method', method, 'run', run + 1)

            tabular_dyna_q = TabularDynaQ(maze, params)
            rewards_new = tabular_dyna_q.resolve_maze(method)
            rewards += rewards_new

        tabular_dyna_q.sample_episode(method)

        rewards /= params.runs

        results.append((rewards, method))

    show_results(results)
    plt.show()

gridworld_experiment()
