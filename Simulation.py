import sys

from our_agent import Multi_QAgent
from our_env import *
import matplotlib.pyplot as plt
'''
This program generates a network, teaches a Q-learning agent
to route packets, and tests both the learned Q-routing policy
and Shortest Path for routing on the network over various
network loads.
'''

'''One episode starts with initialization of all the packets and ends with delivery of all of
env.npackets + env.max_initializations packets OR after time_steps.'''
numEpisode = 40
'''Max length of one episode'''
time_steps = 2000
'''Specify learn method'''
learn_method = 'Multi-Q-learning'
'''Specify reward function (listed in our_env.py)'''
rewardfunction = 'reward1'
'''Mark true to generate plots of performance while learning'''
learning_plot = True
'''Mark true to generate plots of performance for different test network loads'''
comparison_plots = True
'''Number of times to repeat each value in network_load list'''
trials = 10
'''Mark true to perform shortest path simultaneously during testing for comparison to Q-learning'''
sp = False
'''Initialize environment'''
env = dynetworkEnv()
'''Specify list of network loads to test'''
network_load = np.arange(2000, 2500, 500)
print("Network load for test: ", network_load)
for i in network_load:
    if i <= 0:
        print("Error: Network load must be positive.")
        exit()
    if i >= env.nnodes*env.max_queue:
        print("Error: Network load cannot exceed nodes times max queue size.")
env.reset(max(network_load), False)
if learn_method == "Q-learning":
    agent = QAgent(env.dynetwork)
elif learn_method == "Multi-Q-learning":
    agent = Multi_QAgent(env.dynetwork)
else:
    print("No assigned algorithm")
    sys.exit()
print("Algorithm is ", learn_method)

'''Performance Measures for Q-Learning While Learning'''
avg_deliv_learning = []
avg_q_len_learning = []
delivery_ratio = []

# In each episode, update the network for time_steps times. In each time step, update the whole network, which means
# 1.update edges 2.generate packet 3.update queue 4. update packet time in queue 5.route all nodes.
'''----------------------LEARNING PROCESS--------------------------'''
for i_episode in range(numEpisode):
    print("---------- Episode:", i_episode+1," ----------")
    step = []
    deliveries = []
    '''iterate each time step try to finish routing within time_steps'''
    for t in range(time_steps):
        '''key function that obtain action and update Q-table'''
        env.updateWhole(agent, rewardfun=rewardfunction)

        '''store atributes for performance measures'''
        step.append(t)
        deliveries.append(copy.deepcopy(env.dynetwork._deliveries))

        if (env.dynetwork._deliveries >= (env.npackets + env.dynetwork._max_initializations)):
            print("done!")
            break
        
    '''Save all performance measures'''
    avg_deliv_learning.append(env.calc_avg_delivery())
    avg_q_len_learning.append(np.average(env.dynetwork._avg_q_len_arr))
    delivery_ratio.append(env.dynetwork._deliveries/max(network_load))
    print("end to end delay: ", env.calc_avg_delivery())
    print("delivery ratio: ", env.dynetwork._deliveries/max(network_load))
    print("average queue length: ", np.average(env.dynetwork._avg_q_len_arr))
    
    env.reset(max(network_load), False)  # Use the max network load to learn

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'plots/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
learn_results_dir = os.path.join(script_dir, 'plots/learnRes/')
if not os.path.isdir(learn_results_dir):
    os.makedirs(learn_results_dir)

'''Produces plots  while learning '''
print("**********Learning result per episode**********")
if learning_plot:
    print("Average Delivery Time")
    print(avg_deliv_learning)
    plt.clf()
    plt.title("Average Delivery Time Per Episode")
    plt.plot(list(range(1, numEpisode + 1)), avg_deliv_learning)
    plt.xlabel('Episode')
    plt.ylabel('Delay')
    plt.savefig(learn_results_dir + "delay.png")
    np.save("avg_deliv_learning", avg_deliv_learning)
    plt.clf()

    print("Average Queue Length")
    print(avg_q_len_learning)
    plt.clf()
    plt.title("Average Num of Pkts a Node Hold Per Episode")
    plt.plot(list(range(1, numEpisode + 1)), avg_q_len_learning)
    plt.xlabel('Episode')
    plt.ylabel('Average Number of Packets being hold by a Node')
    plt.savefig(learn_results_dir + "avg_q_len_learning.png")
    np.save("avg_q_len_learning", avg_q_len_learning)
    plt.clf() 

    print("Delivery ratio")
    print(delivery_ratio)
    plt.clf()
    plt.title("Delivery Ratio Per Episode")
    plt.plot(list(range(1, numEpisode + 1)), delivery_ratio)
    plt.xlabel("Episode")
    plt.ylabel("Delivery Ratio")
    plt.savefig(learn_results_dir + "delivery_ratio.png")
    np.save("delivery_ratio", delivery_ratio)
    plt.clf()
print("**********End of learning result**********")


'''--------------------------TESTING PROCESS--------------------------'''
'''Performance Measures for Q-Learning'''
avg_deliv = []
avg_q_len = []
delivery_ratio = []

for i in range(len(network_load)):
    curLoad = network_load[i]
    
    print("---------- Testing Load of ", curLoad," ----------")
    for currTrial in range(trials):
        env.reset(curLoad, True)

        step = []
        deliveries = []
    
        '''iterate each time step try to finish routing within time_steps'''
        for t in range(time_steps):
    
            total = env.npackets + env.dynetwork._max_initializations
            q_done = (env.dynetwork._deliveries >= total)
            if sp:
                s_done = (env.dynetwork.sp_deliveries >= total)
            else:
                s_done = True
            env.updateWhole(agent, q=not q_done, sp=not s_done, rewardfun=rewardfunction, savesteps=False)

            if q_done and s_done:
                print("Finished trial ", currTrial)
                break
        
        '''STATS MEASURES'''
        avg_deliv.append(env.calc_avg_delivery())
        avg_q_len.append(np.average(env.dynetwork._avg_q_len_arr))
        delivery_ratio.append(env.dynetwork._deliveries/curLoad)

'''Plot of test'''
print("**********Test result**********")
print("Delay")
print(avg_deliv)
print("Average queue length")
print(avg_q_len)
print("Delivery ratio")
print(delivery_ratio)
print("**********End of test result**********")
