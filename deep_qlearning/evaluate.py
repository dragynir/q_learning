import numpy as np
from gym import Env
from tqdm import tqdm

from deep_qlearning.model import DQN


def evaluate_agent(env: Env, policy_net: DQN, max_steps: int, n_episodes: int):
    """Evaluate the agent for ``n_episodes`` episodes and returns average reward and std of reward.

    :param env: The evaluation environment
    :param policy_net: trained policy network
    :param n_episodes: Number of episode to evaluate the agent
    :param max_steps: max_steps in episode
    :param policy_net: policy network to choose actions
    """
    print('Evaluation...')

    episode_rewards = []
    for episode in tqdm(range(n_episodes), total=n_episodes):
        state, info = env.reset()

        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):

            # Take the action (index) that have the maximum expected future reward given that state
            action = policy_net(state).max(1)[1].item()
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break

            state = new_state

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    print(f'Max possible reward: ', max_steps)  # for cart-pole

    return mean_reward, std_reward