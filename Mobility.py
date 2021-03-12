from pymobility.models.mobility import random_waypoint
import copy

'''This file add mobility to the nodes of the network'''


class Mobility(object):
    def __init__(self, mobility_model, nnodes, min_speed, max_speed):
        self.trajectory = {}  # record trajectory of all nodes.
        self.step = 0    # record steps
        self.node_number = nnodes
        if mobility_model == 'random_waypoint':  # instance of mobility model, actually a iter generator
            self.mb = random_waypoint(nnodes, dimensions=(1, 1), velocity=(min_speed, max_speed), wt_max=1.0)
        else:
            print('Undefined mobility model')

    def get_next_way_point(self):
        positions = next(self.mb)

        # record trajectory of nodes
        for i in range(self.node_number):
            self.trajectory.setdefault(i, []).append(positions[i])

        return positions

    def print_trajectory(self):
        print(self.trajectory)