# -*- coding: utf-8 -*-
from torch import nn


class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    self.state_size = observation_space.shape[0]
    self.action_size = action_space.n

    self.relu = nn.ReLU(inplace=True)
    self.softmax = nn.Softmax()

    # Pass state into model body
    self.conv1 = nn.Conv2d(3, 16, 8, stride=4, padding=3)
    self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
    self.fc1 = nn.Linear(3200, hidden_size)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, self.action_size)
    self.fc_critic = nn.Linear(hidden_size, 1)

  def forward(self, x, h):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = self.softmax(self.fc_actor(x)).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    V = self.fc_critic(x)
    return policy, V, (h[0], h[1])
