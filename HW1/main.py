import numpy as np
import matplotlib.pyplot as plt
import random

def random_walk(x):

    dim = np.size(x)
    walk_set = [-1, 1, 0]
    for i in range(dim):
        x[i] = x[i] + np.random.choice(walk_set)

    return x


def epsilon_greedy(x, epsilon, arms):

    r = random.uniform(0, 1)
    i = np.argmax(x)
    column_indexes = list(range(0, arms))

    if r <= epsilon:
        column_indexes.remove(i)
        i = random.choice(column_indexes)
        return i

    else:
        return i

def multi_task(epsilon=0.1, max_iter=10000, tasks=500, arms=10, alpha=0.1):

    rows, cols = tasks, arms

    q = np.array([([0] * arms) for i in range(rows)])
    constQ = np.array([([0] * cols) for i in range(rows)])
    variabQ = np.array([([0] * cols) for i in range(rows)])
    constN = np.array([([0] * cols) for i in range(rows)])
    variabN = np.array([([0] * cols) for i in range(rows)])
    constR = np.zeros(max_iter)
    variabR = np.zeros(max_iter)

    for i in range(max_iter):

        for j in range(tasks):
            task_q = q[j, :]
            task_q = random_walk(task_q)
            q[j, :] = task_q

            task_constQ = constQ[j, :]
            task_constN = constN[j, :]

            action_index_c = epsilon_greedy(task_constQ, epsilon, arms)

            reward_const = q[j, action_index_c]

            constR[i] = constR[i] + reward_const

            task_constQ[action_index_c] = task_constQ[action_index_c] + alpha * (
                        reward_const - task_constQ[action_index_c])
            constQ[j, :] = task_constQ

            task_constN[action_index_c] = task_constN[action_index_c] + 1
            constN[j:] = task_constN

            task_variabQ = variabQ[j, :]
            task_variabN = variabN[j, :]

            action_index_v = epsilon_greedy(task_variabQ, epsilon, arms)

            reward_variab = q[j, action_index_v]

            variabR[i] = variabR[i] + reward_variab

            task_variabN[action_index_v] = task_variabN[action_index_v] + 1
            variabN[j, :] = task_variabN

            if i == 0:
                beta = 1
            else:
                beta = (1 / task_variabN[action_index_v])

            task_variabQ[action_index_v] = task_variabQ[action_index_v] + \
                                           beta * (reward_variab - task_variabQ[action_index_v])
            variabQ[j, :] = task_variabQ

        constR[i] = constR[i] / tasks
        variabR[i] = variabR[i] / tasks

    return constR, variabR

R_c_step, R_v_step = multi_task()

fig = plt.figure(figsize=(10,5))
fig.add_subplot(111)
plt.xlabel('steps')
plt.ylabel('average reward')
plt.plot(R_c_step, 'r', label='constant stepsize')
plt.plot(R_v_step, 'g', label='varying stepsize')
plt.legend(loc='upper left')
plt.show()
