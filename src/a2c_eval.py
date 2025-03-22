from stable_baselines3 import A2C

from MazeEnv import MazeEnv

if __name__ == '__main__':
    env = MazeEnv()
    model = A2C.load('models/A2C/1000000', env)
    episodes = 10

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()
