import os
import pickle

import numpy as np

from MazeEnv import MazeEnv


def load_q_table():
    filename = 'models/Q/q_table.pkl'
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Q-table file not found: {filename}")

    with open(filename, 'rb') as f:
        q_table = pickle.load(f)

    return q_table


def get_player_pos(obs):
    y, x = np.argwhere(obs == 3)[0]
    return x * obs.shape[0] + y


if __name__ == '__main__':
    env = MazeEnv()
    Q = load_q_table()
    episodes = 10

    for _ in range(episodes):
        state, _ = env.reset()
        state = get_player_pos(state)
        done = False

        while not done:
            env.render()
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            state = get_player_pos(state)
            done = terminated or truncated

    env.close()
