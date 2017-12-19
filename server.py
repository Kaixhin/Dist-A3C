# -*- coding: utf-8 -*-
import argparse
import os
import zmq
import gym
import torch
from torch import optim
from torch import multiprocessing as mp
from torch.autograd import Variable

from model import ActorCritic
from test import test
from utils import send_tensors, receive_tensors


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--port', type=int, default=5556, help='Server port')
parser.add_argument('--env', type=str, default='Pong', help='ATARI game')
parser.add_argument('--num-processes', type=int, default=4, metavar='N', help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=100000000, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=100, metavar='STEPS', help='Max number of forward steps for A3C before update')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='LENGTH', help='Maximum episode length')  # Gym default
parser.add_argument('--hidden-size', type=int, default=256, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
parser.add_argument('--lr', type=float, default=0.0007, metavar='η', help='Learning rate')
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


# Adjusts learning rate
def _adjust_learning_rate(optimiser, lr):
  for param_group in optimiser.param_groups:
    param_group['lr'] = lr


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
  T = 0  # Global counter

  # Server setup
  context = zmq.Context()
  socket = context.socket(zmq.REP)
  socket.bind("tcp://*:%s" % args.port)

  # Create network
  env = gym.make(args.env)
  model = ActorCritic(env.observation_space, env.action_space, args.hidden_size)
  if args.model and os.path.isfile(args.model):
    # Load pretrained weights
    model.load_state_dict(torch.load(args.model))
  # Create optimiser
  optimiser = optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.rmsprop_decay, eps=args.rmsprop_epsilon)
  env.close()

  if args.evaluate:
    # TODO: Run one evaluation process
    # TODO: Change test to run on demand
    pass
  else:
    # Start server
    while T < args.T_max:
      gradients = receive_tensors(socket)  # Receive message from client
      if len(gradients) == 0:
        # If no gradients, send model parameters
        send_tensors(socket, [param.data for param in model.parameters()])
      else:
        # If gradients received, apply gradients to model
        T += int(gradients.pop()[0])  # Increment counter by number of steps
        for param, grad in zip(model.parameters(), gradients):
          if param.grad is not None:
            break
          param._grad = Variable(grad)
        optimiser.step()  # Optimise
        if not args.no_lr_decay:
          _adjust_learning_rate(optimiser, max(args.lr * (args.T_max - T) / args.T_max, 1e-32))  # Linearly decay learning rate
        send_tensors(socket, [])  # Send empty signal back to complete request-response
        print(T)
  """
  # Start validation agent
  processes = []
  p = mp.Process(target=test, args=(0, args, T, shared_model))
  p.start()
  processes.append(p)

  if not args.evaluate:
    # Start training agents
    for rank in range(1, args.num_processes + 1):
      p = mp.Process(target=train, args=(rank, args, T, shared_model, optimiser))
      p.start()
      processes.append(p)

  # Clean up
  for p in processes:
    p.join()
  """
