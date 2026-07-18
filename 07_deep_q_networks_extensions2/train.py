"""Training loop and evaluation helper for the Dueling DQN agent."""

import numpy as np
import gymnasium as gym

from dqn_agent import DuelingDQNAgent


def train_dueling_dqn(
    env_name="LunarLander-v3",
    num_episodes=600,
    max_steps=1000,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    seed=314,
    **agent_kwargs,
):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DuelingDQNAgent(state_size, action_size, **agent_kwargs)

    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        ep_reward = 0.0

        for _ in range(max_steps):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            ep_reward += reward

            if done:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(ep_reward)

        if (episode + 1) % 20 == 0:
            avg_last_20 = np.mean(episode_rewards[-20:])
            print(f"episode {episode + 1}/{num_episodes} | avg reward (last 20): {avg_last_20:.1f} | epsilon: {epsilon:.3f}")

    env.close()
    return agent, episode_rewards


def run_episodes(agent, env_name="LunarLander-v3", num_episodes=5, epsilon=0.0, seed=1000):
    env = gym.make(env_name)
    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.act(state, epsilon)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        rewards.append(ep_reward)
        print(f"episode {episode + 1}: reward = {ep_reward:.1f}")

    env.close()
    print(f"\nmin: {np.min(rewards):.1f}  mean: {np.mean(rewards):.1f}  max: {np.max(rewards):.1f}")
    return rewards
