Dist-A3C
========
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

**TODO: Have server use mp - one thread for server, one for testing. Keep counter to know once finished. Also be able to send push notifications to kill running clients once counter done.**

Distributed asynchronous advantage actor-critic (A3C) [[1]](#references) with generalised advantage estimation (GAE) [[2]](#references). Run `python server.py <options>` to start the server and `python client.py <options>` for as many clients as wanted.

Requirements
------------

- [Python](https://www.python.org/)
- [PyTorch](http://pytorch.org/)
- [OpenAI Gym](https://gym.openai.com/)
- [Plotly](https://plot.ly/python/)
- [PyZMQ](https://github.com/zeromq/pyzmq)
- [MessagePack](http://msgpack.org/)
- [msgpack-numpy](https://github.com/lebedov/msgpack-numpy)

Acknowledgements
----------------

- [@ikostrikov](https://github.com/ikostrikov) for [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)

References
----------

[1] [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783)  
[2] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)  
