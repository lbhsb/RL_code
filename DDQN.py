
import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import math

GAMMA = 0.999
EPS_DECAY = 200
INITIAL_EPSILON = 0.9
FINAL_EPSILON = 0.001
REPLAY_MEMORY = 10000
BATCH = 128
UPDATE_STEPS = 10
learn_steps = 0
writer = SummaryWriter('logs/ddqn')
begin_learn = False
num_episodes = 3000
episode_reward = 0
steps_done = 0

memory_replay = Memory(REPLAY_MEMORY)
epsilon = INITIAL_EPSILON
onlineQNetwork = QNetwork().to(device)
targetQNetwork = QNetwork().to(device)
targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-3)

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(n_state, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)

        self.fc3 = nn.Linear(256, n_action)

    def forward(self, state):
        y = self.relu(self.fc1(state))
        y = self.relu(self.fc2(y))
        y = self.fc3(y)

        return y

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


# onlineQNetwork.load_state_dict(torch.load('ddqn-policy.para'))
for epoch in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    for time_steps in range(500):
        env.render()
        p = random.random()
        if p < epsilon:
            action = random.randint(0, 1)
        else:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = onlineQNetwork.select_action(tensor_state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        memory_replay.add((state, next_state, action, reward, done))

        if begin_learn is False:
            print('learn begin!')
            begin_learn = True
        learn_steps += 1
        if learn_steps % UPDATE_STEPS == 0:
            targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
        batch = memory_replay.sample(BATCH, False)
        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

        batch_state = torch.FloatTensor(batch_state).to(device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(device)
        batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
        batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

        with torch.no_grad():
            onlineQ_next = onlineQNetwork(batch_next_state)
            targetQ_next = targetQNetwork(batch_next_state)
            online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
            y = batch_reward + (1 - batch_done) * GAMMA * targetQ_next.gather(1, online_max_action.long())

        loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # writer.add_scalar('loss', loss.item(), global_step=learn_steps)
        steps_done += 1
        epsilon = FINAL_EPSILON + (INITIAL_EPSILON - FINAL_EPSILON) * \
                  math.exp(-1. * steps_done / EPS_DECAY)

        if done:
            break
        state = next_state
    writer.add_scalar('episode reward', episode_reward, global_step=epoch)
    if epoch % 10 == 0:
        # torch.save(onlineQNetwork.state_dict(), 'ddqn-policy.para')
        print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))

env.close()
