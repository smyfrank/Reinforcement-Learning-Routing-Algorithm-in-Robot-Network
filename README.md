# Reinforcement-Learning-Routing-Algorithm-in-Robot-Network

## Introduction

We implement a simulation of a mobile robot network routing protocol based on multi-agent reinforcement learning.

A mobile robot network is a kind of mobile ad-hoc network that connects mobile robots together. This project simulates the packet routing behavior in the network. The network randomly generates packets to perform routing task which is driven by multi-agent reinforcement learning routing algorithm. The antenna communication range, mobility model, moving speed, node number, packet number, cache queue length, et. can be set manually. 

## Dependence

- Python 3.7
- NetworkX
- Matplotlab
- OpenAI Gym
- numpy
- pymobility

## Code Structure

- our_agent: routing algorithm based on multi-agent reinforcement learning.
- out_env: the simulation environment.
- mobility: define mobility patterns of the robot nodes.
- dynetwork: dynamically drives the network to route packets.
- packet: define packets parameters.
- simulation: set the learning and testing stage; collect result.
