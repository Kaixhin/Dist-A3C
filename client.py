# -*- coding: utf-8 -*-
import argparse
import os
import zmq
import gym
import torch
from torch import nn
from torch.autograd import Variable

from model import ActorCritic
from utils import state_to_tensor
from utils import send_tensors, receive_tensors

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--server', type=str, default='127.0.0.1', help='Server address')
parser.add_argument('--port', type=int, default=5556, help='Server port')
parser.add_argument('--env', type=str, default='Pong', help='ATARI game')
parser.add_argument('--t-max', type=int, default=20, metavar='STEPS', help='Max number of forward steps for A3C before update')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='LENGTH', help='Maximum episode length')  # Gym default
parser.add_argument('--hidden-size', type=int, default=256, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
parser.add_argument('--no-lr-decay', action='store_true', help='Disable linearly decaying learning rate to 0')
parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
parser.add_argument('--rmsprop-epsilon', type=float, default=0.1, metavar='ε', help='RMSprop epsilon')
parser.add_argument('--value-weight', type=float, default=0.5, metavar='WEIGHT', help='Value loss weight')
parser.add_argument('--entropy-weight', type=float, default=0.01, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--max-gradient-norm', type=float, default=40, metavar='VALUE', help='Max value of gradient L2 norm')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=50000, metavar='STEPS', help='Number of training steps between evaluations (roughly)')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')


# Syncs local parameters with server parameters
def _sync_params(socket, model):
  send_tensors(socket, [])
  for param, weight in zip(model.parameters(), receive_tensors(socket)):
    param.data = weight


# Syncs server gradients with local gradients
def _sync_grads(socket, grads):
  send_tensors(socket, grads)
  receive_tensors(socket)


if __name__ == '__main__':
  # BLAS setup
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'

  # Setup
  args = parser.parse_args()
  args.env += 'Deterministic-v4'  # Add ending to ALE game for Gym
  print(' ' * 26 + 'Options')
  for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))

  # Server setup
  context = zmq.Context()
  socket = context.socket(zmq.REQ)
  socket.connect("tcp://%s:%s" % (args.server, args.port))

  # Create network
  env = gym.make(args.env)
  action_size = env.action_space.n
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  model.train()

  t = 1  # Thread step counter
  done = True  # Start new episode

  while True:  # TODO: Need to receive kill signal from server
    # Sync with server model at least every t_max steps
    _sync_params(socket, model)
    # Get starting timestep
    t_start = t

    # Reset or pass on hidden state
    if done:
      hx, cx = Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))
      # Reset environment and done flag
      state = state_to_tensor(env.reset())
      action, reward, done, episode_length = 0, 0, False, 0
    else:
      # Perform truncated backpropagation-through-time (allows freeing buffers after backwards call)
      hx, cx = hx.detach(), cx.detach()

    # Lists of outputs for training
    values, log_probs, rewards, entropies = [], [], [], []

    while not done and t - t_start < args.t_max:
      # Calculate policy and value
      policy, value, (hx, cx) = model(Variable(state), (hx, cx))
      log_policy = policy.log()
      entropy = -(log_policy * policy).sum(1)

      # Sample action
      action = policy.multinomial()
      log_prob = log_policy.gather(1, action.detach())  # Graph broken as loss for stochastic action calculated manually
      action = action.data[0, 0]

      # Step
      state, reward, done, _ = env.step(action)
      state = state_to_tensor(state)
      reward = args.reward_clip and min(max(reward, -1), 1) or reward  # Optionally clamp rewards
      done = done or episode_length >= args.max_episode_length
      episode_length += 1  # Increase episode counter

      # Save outputs for training
      [arr.append(el) for arr, el in zip((values, log_probs, rewards, entropies), (value, log_prob, reward, entropy))]

      # Increment local counter
      t += 1

    # Return R = 0 for terminal s or V(s_i; θ) for non-terminal s
    if done:
      R = Variable(torch.zeros(1, 1))
    else:
      _, R, _ = model(Variable(state), (hx, cx))
    values.append(R.detach())

    # Train the network
    policy_loss = 0
    value_loss = 0
    A_GAE = torch.zeros(1, 1)  # Generalised advantage estimator Ψ
    # Calculate n-step returns in forward view, stepping backwards from the last state
    trajectory_length = len(rewards)
    for i in reversed(range(trajectory_length)):
      # R ← r_i + γR
      R = rewards[i] + args.discount * R
      # Advantage A = R - V(s_i; θ)
      A = R - values[i]
      # dθ ← dθ - ∂A^2/∂θ
      value_loss += 0.5 * A ** 2  # Least squares error

      # TD residual δ = r + γV(s_i+1; θ) - V(s_i; θ)
      td_error = rewards[i] + args.discount * values[i + 1].data - values[i].data
      # Generalised advantage estimator Ψ (roughly of form ∑(γλ)^t∙δ)
      A_GAE = A_GAE * args.discount * args.trace_decay + td_error
      # dθ ← dθ + ∇θ∙log(π(a_i|s_i; θ))∙Ψ
      policy_loss -= log_probs[i] * Variable(A_GAE)  # Policy gradient loss
      # dθ ← dθ + β∙∇θH(π(s_i; θ))
      policy_loss -= args.entropy_weight * entropies[i]  # Entropy maximisation loss

    # Zero local grads
    model.zero_grad()
    # Note that losses were defined as negatives of normal update rules for gradient descent
    (policy_loss + args.value_weight * value_loss).backward()
    # Gradient L2 normalisation
    nn.utils.clip_grad_norm(model.parameters(), args.max_gradient_norm)

    # Transfer gradients to server model and update
    _sync_grads(socket, [param.grad.data for param in model.parameters()] + [torch.Tensor([t - t_start])])  # Also send number of steps

  env.close()
