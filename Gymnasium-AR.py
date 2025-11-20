import gymnasium as gym
import numpy as np

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

def make_inverted_pendulum(render=True, seed=42, theta_limit_deg=24, x_limit=3.0):
    env = gym.make("CartPole-v1", render_mode=("human" if render else None))

    # semente para ambiente e espaço de ações
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    # ajusta limites de ângulo e posição (paredes)
    env.unwrapped.theta_threshold_radians = np.deg2rad(theta_limit_deg)
    env.unwrapped.x_threshold = float(x_limit)

    return env


# Q-learnign com discretização de estados
class Discretizer:
    def __init__(self, env, bins=(6, 6, 12, 12)):
        self.env = env
        self.bins = bins

        obs_space = env.observation_space
        self.low = obs_space.low
        self.high = obs_space.high

        # CartPole tem limites em algumas coordenadas
        self.low[1] = -3.0   # velocidade do carrinho
        self.high[1] = 3.0
        self.low[3] = -5.0   # velocidade angular
        self.high[3] = 5.0

        self.grid = [
            np.linspace(self.low[i], self.high[i], bins[i] - 1)
            for i in range(len(bins))
        ]

    def discretize(self, obs):
        idxs = []
        for i, g in enumerate(self.grid):
            idx = np.digitize(obs[i], g)
            idxs.append(idx)
        return tuple(idxs)

def train_q_learning(
    num_episodes=5000,
    alpha=0.1, # Taxa de aprendizado
    gamma=0.99, # Fator de desconto
    epsilon_start=1.0, # Decrescimento exponencial epsilon
    epsilon_end=0.01,
    epsilon_decay=0.995,
):
    env = make_inverted_pendulum(render=False)

    # Instancia a discretização do espaço de estados
    disc = Discretizer(env)

    n_actions = env.action_space.n 
    
    # Criação da Q-table
    q_shape = disc.bins + (n_actions,)
    Q = np.zeros(q_shape)

    epsilon = epsilon_start
    rewards_per_episode = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        state = disc.discretize(obs)
        done = False
        ep_reward = 0.0

        while not done:
            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = disc.discretize(next_obs)

            best_next = np.max(Q[next_state])
            td_target = reward + gamma * best_next * (not done)
            td_error = td_target - Q[state + (action,)]
            Q[state + (action,)] += alpha * td_error

            state = next_state
            ep_reward += reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(ep_reward)

        if (ep + 1) % 100 == 0:
            print(f"[Q-Learning] Episódio {ep+1}, recompensa média ult.100 = "
                  f"{np.mean(rewards_per_episode[-100:]):.1f}")

    env.close()
    return Q, rewards_per_episode

#Deep Q-Network

class DQNNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)
class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*batch)
        )
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

def train_dqn(
    num_episodes=800,
    gamma=0.99, # Taxa de redução
    lr=1e-3, # Taxa de aprendizado
    batch_size=64,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    target_update_freq=10, # A target network é atualizada a cada 10 passos (episódios)
):
    env = make_inverted_pendulum(render=False)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Criação das duas Networks
    policy_net = DQNNet(obs_dim, n_actions).to(device)
    target_net = DQNNet(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer()

    epsilon = epsilon_start
    rewards_per_episode = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            # epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    q_values = policy_net(state_t)
                    action = int(torch.argmax(q_values, dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward

            # atualização da rede
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                q_values = policy_net(states_t).gather(1, actions_t)

                with torch.no_grad():
                    next_q_values = target_net(next_states_t).max(dim=1, keepdim=True)[0]
                    target = rewards_t + gamma * next_q_values * (1 - dones_t)

                loss = nn.functional.mse_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_per_episode.append(ep_reward)

        # atualiza target net de tempos em tempos
        if (ep + 1) % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (ep + 1) % 20 == 0:
            print(f"[DQN] Episódio {ep+1}, recompensa média ult.20 = "
                  f"{np.mean(rewards_per_episode[-20:]):.1f}, epsilon = {epsilon:.3f}")

    env.close()
    return policy_net, rewards_per_episode

def plot_rewards(rewards, title, filename):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()

if __name__ == "__main__":
    _, q_rewards = train_q_learning()
    plot_rewards(q_rewards, "Q-Learning: Recompensa por episódio", "q_learning_rewards.png")

    _, dqn_rewards = train_dqn()
    plot_rewards(dqn_rewards, "DQN: Recompensa por episódio", "dqn_rewards.png")
