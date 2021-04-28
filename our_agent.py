import collections
import numpy as np
import random
import networkx as nx

'''
    The agent file defines a learning agent and its hyperparameters
    Q-table is a independent data structure which is not stored in each node.
    File contains functions:
        generate_q_table: initialize Q-table
        act: returns which next node to send packet to
        learn: update Q-table after receiving corresponding rewards
'''
class QAgent(object):

    def __init__(self, dynetwork):
        """ 
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy 
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values 
        
        """
        self.config = {
            "learning_rate": 0.8,
            "epsilon": 0.6,
            "discount": 0.9,
            "decay_rate": 0.999,
            "update_epsilon": False,
            }
        self.q = self.generate_q_table(dynetwork._network)
        (self.strategy_table, self.average_strategy_table) = self.generate_strategy_table(dynetwork._network)

    ''' Use this function to initialize the q-table, the q-table is stable since the network is not mobile'''
    # TODO: q-table must be generated that can be adjust to mobile network.
    def generate_q_table(self, network):
        print("Begin to generate_q_table")
        q_table = {}
        num_nodes = network.number_of_nodes()
        for currpos in range(num_nodes):
            nlist = list(range(num_nodes))
            for dest in range(num_nodes):
                q_table[(currpos, dest)] = {}
                for action in nlist:
                    if currpos != dest:
                        ''' Initialize 0 Q-table except destination '''
                        q_table[(currpos, dest)][action] = 0
                        ''' Initialize using Shortest Path'''
                    else:
                        # TODO: Initialize q_table value when current node is destination
                        q_table[(currpos, dest)][action] = 10  # Why set 10 if current node is destination?
        print("q-table:")
        print(q_table)
        print("End of generate_q_table")
        return q_table

    '''Initialize the strategy-table'''
    def generate_strategy_table(self, network):
        print("Begin to generate_strategy_table")
        strategy_table = {}
        average_strategy_table = {}
        num_nodes = network.number_of_nodes()
        for currpos in range(num_nodes):
            nlist = list(range(num_nodes))
            for dest in range(num_nodes):
                strategy_table[(currpos, dest)] = {}
                average_strategy_table[(currpos, dest)] = {}
                for action in nlist:
                    if currpos != dest:
                        '''Initialize 1/|A| in strategy-table and average strategy table except destination'''
                        strategy_table[(currpos, dest)][action] = 1 / (network.number_of_nodes() - 1)
                        average_strategy_table[(currpos, dest)][action] = 1 / (network.number_of_nodes() - 1)
                    else:
                        strategy_table[(currpos, dest)][action] = 0
                        average_strategy_table[(currpos, dest)][action] = 0
        print("strategy-table:")
        print(strategy_table)
        print("average-strategy-table:")
        print(average_strategy_table)
        print("End of generate_strategy_table")
        return strategy_table, average_strategy_table

    '''Returns best action for a given state, action is the next step node number. '''
    def act(self, state, neighbor):
        ''' We will either random explore or refer to Q-table with probability epsilon '''
        if random.uniform(0, 1) < self.config['epsilon']:
            """ checks if the packet's current node has any available neighbors """
            if not bool(neighbor):  # In python, blank {}, [], () are all False
                return None
            else:
                next_step = random.choice(neighbor)  # Explore action space
        else:
            temp_neighbor_dict = {n: self.q[state][n] for n in self.q[state] if n in neighbor}  # { expression for x in X [if condition] for y in Y [if condition]...}
            """ checks if the packet's current node has any available neighbors """
            if not bool(temp_neighbor_dict):
                return None
            else:
                next_step = max(temp_neighbor_dict, key=temp_neighbor_dict.get)
                if self.config['update_epsilon']:
                    self.config['epsilon'] = self.config["decay_rate"] * self.config['epsilon']
                    self.config['update_epsilon'] = False
        return next_step

    """updates q-table given current state, reward, and action where a state is a (Node, destination) pair and an action is a step to of the neighbors of the Node """
    def learn(self, current_event, reward, action):
        if (action == None) or (reward == None):
            pass
        else:
            n = current_event[0]
            dest = current_event[1]
            max_q = max(self.q[(action, dest)].values())  # change to max if necessary

            """ Q learning algorithm """
            self.q[(n, dest)][action] = self.q[(n, dest)][action] + (self.config["learning_rate"])*(reward + self.config["discount"] * max_q - self.q[(n, dest)][action])
