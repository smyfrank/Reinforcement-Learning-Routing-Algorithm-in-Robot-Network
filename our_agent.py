import sys

import numpy as np
import random

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
            "learning_rate": 0.3,
            "epsilon": 0.3,
            "discount": 0.9,
            "decay_rate": 0.999,
            "update_epsilon": False,
            }
        self.number_of_nodes = dynetwork._network.number_of_nodes()
        self.q = self.generate_q_table(dynetwork._network)

    ''' Use this function to initialize the q-table, the q-table is stable since the network is not mobile'''
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
        print("End of generate_q_table")
        return q_table

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


"""Class Multi_QAgent inherit class QAgent to perform multi-agent reinforcement learning"""
class Multi_QAgent(QAgent):

    def __init__(self, dynetwork):
        QAgent.__init__(self, dynetwork)
        self.config = {
            "learning_rate": 0.3,
            "epsilon": 0.3,
            "discount": 0.9,
            "decay_rate": 0.999,
            "update_epsilon": False,
            "delta_win": 0.0025,
            "delta_lose": 0.01
            }
        (self.policy, self.mean_policy) = self.generate_strategy_table(dynetwork._network)
        self.counter = self.generate_counter()
        (self.old_neighbors, self.new_neighbors) = self.generate_neighbor_table()

    """Initialize the counter"""
    def generate_counter(self):
        print("Begin to generate counter")
        counter = {}
        for currpos in range(self.number_of_nodes):
            nlist = list(range(self.number_of_nodes))
            for dest in range(self.number_of_nodes):
                if currpos != dest:
                    counter[(currpos, dest)] = 0
        print("End of generate counter")
        return counter

    """Initialize the strategy-table"""
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
                    if (currpos != dest) and (currpos != action):
                        '''Initialize 1/|A| in strategy-table and average strategy table except destination'''
                        strategy_table[(currpos, dest)][action] = 1 / (network.number_of_nodes() - 1)
                        average_strategy_table[(currpos, dest)][action] = 1 / (network.number_of_nodes() - 1)
        print("End of generate_strategy_table")
        return strategy_table, average_strategy_table

    """Initialize neighbors history records"""
    def generate_neighbor_table(self):
        print("Begin to generate old and new neighbor table")
        new_neighbor_table = dict.fromkeys(range(self.number_of_nodes), set())
        old_neighbor_table = dict.fromkeys(range(self.number_of_nodes), set())
        print("new_neighbor_table:", new_neighbor_table)
        print("old_neighbor_table:", old_neighbor_table)
        return old_neighbor_table, new_neighbor_table

    """Returns best action for a given state, action is the next step node number, depends on exploration-exploitation , strategy-table and q-table"""
    def act(self, state, neighbor):
        # TODO: exploration and exploitation
        if random.uniform(0, 1) < self.neighbors_variation(state, neighbor) * self.config['epsilon'] + 0.3:
            if not bool(neighbor):
                return None
            else:
                next_step = random.choice(neighbor)
                return next_step
        else:
            # TODO: policy_value_list may all be 0
            temp_neighbor_strategy_dict = {n: self.policy[state][n] for n in self.policy[state] if n in neighbor}
            policy_key_list = list(temp_neighbor_strategy_dict.keys())
            policy_value_list = list(temp_neighbor_strategy_dict.values())
            if not bool(temp_neighbor_strategy_dict):
                return None
            else:
                try:
                    if np.sum(policy_value_list) == 1.0:
                        next_step = np.random.choice(a=policy_key_list, p=policy_value_list)
                        return next_step
                    else:
                        if not any(policy_value_list):
                            next_step = random.choice(neighbor)
                            return next_step
                        else:
                            policy_value_list_new = np.divide(policy_value_list, sum(policy_value_list))
                            next_step = np.random.choice(policy_key_list, p=policy_value_list_new)
                            return next_step
                except ValueError:
                    print("Value Error, " + str(policy_value_list))
                    sys.exit()

    """update q-table, policy, mean-policy"""
    def learn(self, current_event, reward, action):
        if (action == None) or (reward == None):
            pass
        else:
            cur_pos = current_event[0]
            dest = current_event[1]
            max_q = max(self.q[(action, dest)].values())

            """update q-table"""
            self.q[(cur_pos, dest)][action] = self.q[(cur_pos, dest)][action] + (self.config["learning_rate"])*(reward + self.config["discount"] * max_q - self.q[(cur_pos, dest)][action])

            """update mean-policy"""
            self.update_mean_pi(current_event)

            """update policy"""
            self.update_pi(current_event)
        return

    """calculate delta"""
    def delta(self, state):
        sum_policy = 0.0
        sum_mean_policy = 0.0
        for i in self.policy[(state[0], state[1])].keys():
            sum_policy += (self.policy[state[0], state[1]][i] * self.q[state[0], state[1]][i])
            sum_mean_policy += (self.mean_policy[state[0], state[1]][i] * self.q[state[0], state[1]][i])
        if (sum_policy > sum_mean_policy):
            return self.config["delta_win"]
        else:
            return self.config["delta_lose"]

    """update policy table"""
    def update_pi(self, state):
        maxQValueIndex = max(self.q[(state[0], state[1])], key=self.q[(state[0], state[1])].get)
        for i in self.policy[(state[0], state[1])].keys():
            d_plus = self.delta(state)
            d_minus = ((-1.0) * d_plus) / ((self.number_of_nodes - 1) - 1.0)
            if (i == maxQValueIndex):
                self.policy[(state[0], state[1])][i] = min(1.0, self.policy[(state[0], state[1])][i] + d_plus)
            else:
                self.policy[(state[0], state[1])][i] = max(0.0, self.policy[(state[0], state[1])][i] + d_minus)
        return

    """update mean-policy table"""
    def update_mean_pi(self, state):
        self.counter[(state[0], state[1])] += 1
        for i in self.policy[(state[0], state[1])].keys():
            self.mean_policy[(state[0], state[1])][i] += ((1.0/self.counter[(state[0], state[1])]) * (self.policy[(state[0], state[1])][i]) - self.mean_policy[(state[0], state[1])][i])
        return

    """calculate the variation of the number of neighbor nodes"""
    def neighbors_variation(self, state, new_neighbors):
        cur_node = state[0]
        self.old_neighbors[cur_node] = self.new_neighbors[cur_node]
        self.new_neighbors[cur_node] = set(new_neighbors)
        union = len(self.new_neighbors[cur_node].union(self.old_neighbors[cur_node]))
        if union == 0:
            return 1.0
        inter = len(self.new_neighbors[cur_node].intersection(self.old_neighbors[cur_node]))
        return (union - inter) / union
