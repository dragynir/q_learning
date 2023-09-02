import time
from random import random

import imageio
import numpy as np
from gym import Env
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2

def initialize_q_table(state_space, action_space):
    Qtable = np.zeros((state_space, action_space))
    return Qtable


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state])

    return action


def epsilon_greedy_policy(env, Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = np.random.uniform(0, 1)
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(Qtable, state)
        # else --> exploration
    else:
        action = env.action_space.sample() # Take a random action

    return action


def train_qtable(env, Qtable, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, gamma, learning_rate):

    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # Reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        show_env_example(env)

        # repeat
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(env, Qtable, state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

            # If terminated or truncated finish the episode
            if terminated or truncated:
                break

            # Our next state is the new state
            state = new_state

    return Qtable

def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()

        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def record_video(env, Qtable, out_directory, max_steps, fps=1):
    """
    Generate a replay video of the agent
    :param env
    :param Qtable: Qtable of our agent
    :param out_directory
    :param max_steps
    :param fps: how many frame per seconds (with taxi-v3 and frozenlake-v1 we use 1)
    """
    images = []
    terminated = False
    truncated = False
    state, info = env.reset(seed=np.random.randint(0, 500))
    img = env.render()
    images.append(img)
    print('Recording video...')

    steps = 0

    with tqdm(total=max_steps) as pbar:

        while not terminated or truncated:
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(Qtable[state][:])
            state, reward, terminated, truncated, info = env.step(action) # We directly put next_state = state for recording logic
            img = env.render()
            images.append(img)

            if terminated or truncated:
                break

            steps += 1
            pbar.update(1)
            if steps == max_steps:
                break

    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps)


def show_video(path: str, sleep_frame=1) -> None:
    cap = cv2.VideoCapture(path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('window-name', frame)
        count = count + 1

        time.sleep(sleep_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # destroy all opened windows


def show_env_example(env: Env):
    env.reset()
    plt.figure()
    plt.imshow(env.render())
    plt.show()
