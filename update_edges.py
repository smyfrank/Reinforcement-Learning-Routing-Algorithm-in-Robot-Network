import numpy as np
import random
import math

''' Functions to handle edges in our network. '''

''' Randomly deletes some number of edges between min_edge_removal and max_edge_removal '''
def Delete(dyNetwork, min_edge_removal, max_edge_removal):
    edges = dyNetwork._network.edges()
    deletion_number = random.randint(min_edge_removal, min(max_edge_removal, len(edges) - 1))
    strip = random.sample(edges, k=deletion_number)
    temp = []
    for s_edge, e_edge in strip:
        temp.append((s_edge,e_edge,dyNetwork._network[s_edge][e_edge]))
    strip = temp
    dyNetwork._network.remove_edges_from(strip)
    dyNetwork._stripped_list.extend(strip)


''' Randomly restores some edges we have deleted '''
def Restore(dyNetwork):
    restore_number = random.randint(0, len(dyNetwork._stripped_list))
    restore = random.choices(dyNetwork._stripped_list, k=restore_number)
    dyNetwork._network.add_edges_from(restore)


''' Randomly change edge weights '''
# edge weight is actually edge delay
def Random_Walk(dyNetwork):
    for s_edge, e_edge in dyNetwork._network.edges():
        try:
            changed = random.randint(-2, 2) + dyNetwork._network[s_edge][e_edge]['edge_delay']
            dyNetwork._network[s_edge][e_edge]['edge_delay'] = max(changed, 1)
        except:
            print(s_edge, e_edge)


''' Change edge weights so that the edge weight changes will be roughly sinusoidal across the simulation '''
# Why sine_state step is pi/6? Maybe it is just a super parameter.
def Sinusoidal(dyNetwork):
    for s_edge, e_edge in dyNetwork._network.edges():
        dyNetwork._network[s_edge][e_edge]['edge_delay'] = max(1, int(dyNetwork._network[s_edge][e_edge]['initial_weight']* (1 + 0.5 * math.sin(dyNetwork._network[s_edge][e_edge]['sine_state']))))
        dyNetwork._network[s_edge][e_edge]['sine_state'] += math.pi/6


''' Not in use. If it were used the edge weight would be the average of the number of packets in each queue of the endpoints of the edge. '''
def Average(dyNetwork):
    for node1, node2 in dyNetwork._network.edges(data = False):
        tot_queue1 = dyNetwork._network.nodes[node1]['sending_queue']
        tot_queue2 = dyNetwork._network.nodes[node2]['sending_queue']
        avg = np.avg([tot_queue1, tot_queue2])
        dyNetwork._network[node1][node2]['edge_delay'] = avg
        del tot_queue1, tot_queue2


''' Judge whether two nodes are in connection depending on the positions and communication radius'''
def calculate_nodes_connection(dyNetwork, radius):
    for i in range(dyNetwork._network.number_of_nodes()):
        for j in range(i, dyNetwork._network.number_of_nodes()):
            if i != j:
                pos1 = dyNetwork._network.nodes[i]['pos']
                pos2 = dyNetwork._network.nodes[j]['pos']
                if math.sqrt(math.pow((pos1[0]-pos2[0]), 2) +math.pow((pos1[1]-pos2[1]), 2)) <= radius:
                    dyNetwork._network.add_edge(i, j)

                    # TODO: Here we set the edge delay as 1 temporarily
                    dyNetwork._network[i][j]['edge_delay'] = 1
                else:
                    if dyNetwork._network.has_edge(i, j):
                        dyNetwork._network.remove_edge(i, j)
