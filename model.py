# -*- coding: utf-8 -*-
from torch import nn
from torch.nn import functional as F


class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size):
    super(ActorCritic, self).__init__()
    self.state_size = observation_space.shape[0]
    self.action_size = action_space.n

    # Pass state into model body
    self.conv1 = nn.Conv2d(3, 16, 8, stride=4, padding=3)
    self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
    self.fc1 = nn.Linear(3200, hidden_size)
    self.lstm = nn.LSTMCell(hidden_size, hidden_size)
    self.fc_actor = nn.Linear(hidden_size, self.action_size)
    self.fc_critic = nn.Linear(hidden_size, 1)

  def forward(self, x, h):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = F.softmax(self.fc_actor(x), dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    V = self.fc_critic(x)
    return policy, V, (h[0], h[1])
