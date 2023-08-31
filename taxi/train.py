import numpy as np
import gymnasium as gym
import random
import imageio
import os
import tqdm

from gym import Env
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from taxi.q_learning import initialize_q_table, train_qtable, evaluate_agent, record_video, show_video

from taxi.config import Config


def show_env_example(env: Env):
    env.reset()
    plt.figure()
    plt.imshow(env.render())
    plt.show()


def train():
    """
    Main function for q-learning train.

    Using q-table we overfiting the environment, so  in
    random map we will fail.


    Example to create a custom map
    desc = ['SFFF', 'FHFH', 'FFFH', 'HFFG']
    gym.make('FrozenLake-v1', desc=desc, is_slippery=True)
    """
    config = Config()
    env = gym.make(
        config.env_name,
        render_mode=config.render_mode,
    )

    print('_____OBSERVATION SPACE_____ \n')
    print('Observation Space', env.observation_space)
    print('Sample observation', env.observation_space.sample())  # Get a random observation
    show_env_example(env)

    state_space = env.observation_space.n
    print('There are ', state_space, ' possible states')

    action_space = env.action_space.n
    print('There are ', action_space, ' possible actions')

    Qtable = initialize_q_table(state_space, action_space)

    print('_____TRAINING_____ \n')

    Qtable_frozenlake = train_qtable(
        env,
        Qtable,
        config.n_training_episodes,
        config.min_epsilon,
        config.max_epsilon,
        config.decay_rate,
        config.max_steps,
        config.gamma,
        config.learning_rate,
    )

    # Evaluate our Agent
    eval_seed = []  # [42] * config.n_eval_episodes
    mean_reward, std_reward = evaluate_agent(
        env,
        config.max_steps,
        config.n_eval_episodes,
        Qtable_frozenlake,
        eval_seed,
    )
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    trained_video = './replays/trained.mp4'
    if os.path.exists(trained_video):
        os.remove(trained_video)
    record_video(env, Qtable_frozenlake, out_directory=trained_video, max_steps=config.max_steps, fps=1)
    show_video(trained_video)


if __name__ == '__main__':
    train()
