# -*- coding: utf-8 -*-
import os
from PIL import Image
import plotly
from plotly.graph_objs import Scatter, Line
import msgpack
import msgpack_numpy
import torch
from torch import multiprocessing as mp
from torchvision import transforms


to_tensor = transforms.ToTensor()
# Patch MessagePack to work with numpy arrays
msgpack_numpy.patch()


# Global counter
class Counter():
  def __init__(self):
    self.val = mp.Value('i', 0)
    self.lock = mp.Lock()

  def increment(self):
    with self.lock:
      self.val.value += 1

  def value(self):
    with self.lock:
      return self.val.value


# Sends Torch tensor over ØMQ (via numpy format as torch storage does not provide a buffer interface)
def send_tensors(socket, tensors, flags=0, copy=True, track=False):  # TODO: Investigate options
  return socket.send(msgpack.packb([tensor.numpy() for tensor in tensors]), flags, copy=copy, track=track)


# Receives Torch tensor over ØMQ
def receive_tensors(socket, flags=0, copy=True, track=False):
  msg = socket.recv(flags=flags, copy=copy, track=track)
  return [torch.from_numpy(tensor) for tensor in msgpack.unpackb(msg)]


# Preprocesses ALE frames for A3C
def _preprocess(img):
  return to_tensor(Image.fromarray(img, mode='RGB').resize([84, 84]))


# Converts a state from the OpenAI Gym (a numpy array) to a batch tensor
def state_to_tensor(state):
  return _preprocess(state).unsqueeze(0)


# Plots min, max and mean + standard deviation bars of a population over time
def plot_line(xs, ys_population, path=''):
  max_colour, mean_colour, std_colour = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)'

  ys = torch.Tensor(ys_population)
  ys_min = ys.min(1)[0].squeeze()
  ys_max = ys.max(1)[0].squeeze()
  ys_mean = ys.mean(1).squeeze()
  ys_std = ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color='transparent'), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color='transparent'), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title='Rewards',
                   xaxis={'title': 'Step'},
                   yaxis={'title': 'Average Reward'})
  }, filename=os.path.join(path, 'rewards.html'), auto_open=False)
