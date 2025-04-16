import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Noisy Linear Layer for NoisyNet
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()

        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            NoisyLinear(state_size, 128), nn.ReLU(), NoisyLinear(128, 128), nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 64), nn.ReLU(), NoisyLinear(64, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 64), nn.ReLU(), NoisyLinear(64, action_size)
        )

    def forward(self, x):
        features = self.feature_layer(x)

        # Value stream
        value = self.value_stream(features)

        # Advantage stream
        advantage = self.advantage_stream(features)

        # Combine value and advantage using the dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DQNAgent:
    def __init__(self, state_size, action_size, frame_skip=4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.002
        self.batch_size = 256
        self.memory = deque(maxlen=20000)
        self.model = DuelingDQN(state_size, action_size).to(device)
        self.target_model = DuelingDQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()
        self.frame_skip = frame_skip
        self.early_stop_score = 30
        self.early_stop_count = 0
        self.early_stop_patience = 5

    def update_target_model(self):
        tau = 0.01
        for target_param, param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.model.reset_noise()
        with torch.no_grad():
            act_values = self.model(state)
            return torch.argmax(act_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        # Reward normalization
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        self.model.reset_noise()
        self.target_model.reset_noise()

        # Double DQN: Use online network to select actions, target network to evaluate them
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            # Select actions using online network
            next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
            # Evaluate actions using target network
            next_q_values = (
                self.target_model(next_states).gather(1, next_actions).squeeze()
            )
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=device))
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def should_early_stop(self, score):
        if score >= self.early_stop_score:
            self.early_stop_count += 1
        else:
            self.early_stop_count = 0
        return self.early_stop_count >= self.early_stop_patience
