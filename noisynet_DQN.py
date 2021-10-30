import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from collections import Counter
from collections import deque
import matplotlib.pyplot as plt
from replay_buffer import *
from tensorboardX import SummaryWriter

env_id = "Enduro-ram-v0"
env = gym.make(env_id).unwrapped
writer = SummaryWriter('logs_1/noisy_dqn')
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.uniform_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class NoisyDQN(nn.Module):
    def __init__(self, observation_space, action_sapce):
        super(NoisyDQN, self).__init__()

        self.linear = nn.Linear(observation_space, 64)
        self.noisy1 = NoisyLinear(64, 128)
        self.noisy2 = NoisyLinear(128, 256)
        self.noisy3 = NoisyLinear(256, action_sapce)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy1(x))
        x = F.relu(self.noisy2(x))
        x = self.noisy3(x)
        return x

    def act(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.forward(state)
        action = q_value.max(1)[1].data[0]
        action = action.cpu().numpy()  
        action = int(action)
        return action

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, aciton, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, aciton, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done


def compute_td_loss(current_model, target_model, optimizer, replay_buffer, gamma, batch_size, beta):
    state, action, reward, next_state, done, weights, indices = replay_buffer.sample(batch_size, beta)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(np.float32(done)))
    weights = Variable(torch.FloatTensor(weights))

    q_values = current_model(state)
    next_q_values = target_model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]

    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    current_model.reset_noise()
    target_model.reset_noise()

    return loss


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())  


def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


def main():
    observation_space = env.observation_space.shape[0]
    action_sapce = env.action_space.n

    current_model = NoisyDQN(observation_space, action_sapce)
    target_model = NoisyDQN(observation_space, action_sapce)

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model = target_model.cuda()

    optimizer = optim.Adam(current_model.parameters(),lr=0.001)

    beta_start = 0.4
    beta_frames = 1000
    beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

    replay_buffer = PrioritizedReplayBuffer(10000, alpha=0.6)

    update_target(current_model, target_model)

    num_frames = 500
    batch_size = 128
    gamma = 0.999
    steps = 0
    losses = []
    all_rewards = []
    num_eposides = 3000
    state = env.reset()
    for i in range(num_eposides):
        env.reset()
        episode_reward = 0
        for frame_idx in range(1, num_frames + 1):
            env.render()
            action = current_model.act(state)
            steps += 1
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = env.reset()
                all_rewards.append(episode_reward)
                break

            if len(replay_buffer) > batch_size:
                beta = beta_by_frame(frame_idx)
                loss = compute_td_loss(current_model, target_model, optimizer, replay_buffer, gamma, batch_size, beta)
                # writer.add_scalar('loss', loss.item(), global_step=steps)
                losses.append(np.array(loss.data.cpu()))

            # if frame_idx % 200 == 0:
            #     plot(frame_idx, all_rewards, losses)
        writer.add_scalar('episode reward', episode_reward, global_step=i)
        if i % 10 == 0:
            update_target(current_model, target_model)
            print('Ep {}\tMoving average score: {:.2f}\t'.format(i, episode_reward))


if __name__ == '__main__':
    main()
