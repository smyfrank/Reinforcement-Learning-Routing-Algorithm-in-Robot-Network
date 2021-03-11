from pymobility.models.mobility import random_waypoint

'''This file add mobility to the nodes of the network'''
class Mobility(object):
    def __init__(self, network):
        randomWaypoint = random_waypoint(network.nnodes, dimensions=(1, 1), velocity=(network.minSpeed, network.maxSpeed), wt_max=1.0)
