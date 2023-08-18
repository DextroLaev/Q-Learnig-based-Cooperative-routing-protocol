import numpy as np
import matplotlib.pyplot as plt
import sys
from contextlib import redirect_stdout
import os
from utils import *

class Agent:
    def __init__(self,env,discount_factor,learning_rate):
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        # contains the final Q-values for using in future.
        self.Q_table = np.zeros((self.env.n_states,self.env.n_actions))
        self.total_nodes = self.env.nnodes
        self.non_cooperative_routing_path = None
        self.cooperative_routing_path = None
        self.custom_cooperative_routing_path = None
        self.non_cooperative_energy_consumption = 0
        self.cooperative_energy_consumption = 0
        self.custom_cooperative_energy_consumption = 0

    def select_source(self):
        return self.env.source_node

    def select_action(self):
        return np.random.randint(0,self.env.n_actions)

    def reset_Q_table(self):
        self.Q_table = np.zeros((self.env.n_states,self.env.n_actions))

    def run_agent(self,episodes=100):
        self.reset_Q_table()
        for i in range(episodes):
            done = False
            state = self.select_source()

            while not done:

                self.env.state = state
                action = self.select_action()
                next_state,reward,done = self.env.step(action)
                details = {'state':state,'action':action,'reward':reward,'next_state':next_state,'done':done}
                self.Q_table[state][action] = reward + self.discount_factor*np.max(self.Q_table[next_state,:])            # Q-table is updated using the Q-Learning algorithm
                if done:
                    break
                state = next_state

    def select_routing_path(self):
        if os.path.exists('log.txt'):
            os.remove('log.txt')
        copy_Q_table = self.Q_table.copy()
        self.non_cooperative_routing_path = self.non_cooperative_routing(copy_Q_table)
        self.cooperative_routing_path = self.cooperative_routing()
        self.custom_cooperative_routing_path = self.custom_cooperative_routing()

        # following code is to generate log_file
        with open('log_file.txt', 'a') as f:
            with redirect_stdout(f):
                print("Node locations: {}".format(self.env.node_locations))
                print('Non-Cooperative: {}, energy = {}'.format(self.non_cooperative_routing_path,self.non_cooperative_energy_consumption))
                print('Cooperative: {}, energy = {}'.format(self.cooperative_routing_path,self.cooperative_energy_consumption))
                print('Custom Cooperative: {}, energy = {} '.format(self.custom_cooperative_routing_path,self.custom_cooperative_energy_consumption))
        print('Non-Cooperative: {}, energy = {}'.format(self.non_cooperative_routing_path,self.non_cooperative_energy_consumption))
        print('Cooperative: {}, energy = {}'.format(self.cooperative_routing_path,self.cooperative_energy_consumption))
        print('Custom Cooperative: {}, energy = {} '.format(self.custom_cooperative_routing_path,self.custom_cooperative_energy_consumption))



    def non_cooperative_routing(self,Q_table):
        routing_path = [self.env.source_node]
        while self.env.destination_node not in routing_path:
            prev_node = routing_path[-1]
            Q_table[:,prev_node] = -1000
            node_loc = np.where(Q_table[prev_node]==max(Q_table[prev_node]))
            node_location = self.filter_route_node(node_loc,prev_node)
            distance = self.env.euclidean_distance(self.env.node_locations[prev_node],self.env.node_locations[node_location])
            if node_location in routing_path:
                print('Packet is looping back, Fix the error')
                break
            routing_path.append(node_location)
            self.non_cooperative_energy_consumption += attenuation(distance)
        return routing_path

    def cooperative_routing(self):
        Q_table = self.Q_table.copy()
        routing_path = [self.env.source_node]
        while self.env.destination_node not in routing_path:

            prev_node = routing_path[-1]
            Q_table[:,prev_node] = -1000
            node_loc = np.where(Q_table[prev_node]==max(Q_table[prev_node]))
            node_location = self.filter_route_node(node_loc,prev_node)
            distance_i_j = self.env.euclidean_distance(self.env.node_locations[prev_node],self.env.node_locations[node_location])
            with open('log_file.txt', 'a') as f:
                with redirect_stdout(f):
                    print('---------DEBUG starts here------------')
                    print('prev = {}, curr = {}, dist = {}'.format(prev_node,node_location,distance_i_j))
            if (distance_i_j <= self.env.r1):
                energy = cooperative_energy_consumption(distance_i_j,0,0)
                routing_path.append(node_location)
            elif (distance_i_j > self.env.r1) and (distance_i_j <= self.env.r2):

                nodes_between = self.find_all_nodes_between(prev_node,node_location)
                if len(nodes_between) >= 1:
                    with open('log.txt', 'a') as f:
                        with redirect_stdout(f):
                            print('\n\tNODES ARE THERE IN BETWEEN FOR COOPERATION\n')
                            print("\tAvailable Nodes are: {}".format(nodes_between))
                    min_E = (2**63)-1
                    min_index = 0
                    for i in range(len(nodes_between)):
                        curr_node = nodes_between[i]
                        distance = self.env.euclidean_distance(self.env.node_locations[prev_node],self.env.node_locations[curr_node])
                        energy = attenuation(distance)
                        with open('log_file.txt', 'a') as f:
                            with redirect_stdout(f):
                                print('\tprev = {}, curr = {}, dist = {}, energy = {}'.format(prev_node,curr_node,distance,energy))
                        if energy <= min_E:
                            min_E = energy
                            min_index = i
                    distance_c_j = self.env.euclidean_distance(self.env.node_locations[nodes_between[min_index]],self.env.node_locations[node_location])
                    delta = 1 if distance_i_j > self.env.r1 else 0
                    energy = cooperative_energy_consumption(distance_i_j,distance_c_j,delta)
                    routing_path.append(nodes_between[min_index])
                    with open('log_file.txt', 'a') as f:
                        with redirect_stdout(f):
                            print('\tNode Taken for cooperative: {}, energy = {}'.format(nodes_between[min_index],min_E))
                else:
                    energy = cooperative_energy_consumption(distance_i_j,0,1)
                    routing_path.append(node_location)

            else:
                energy = cooperative_energy_consumption(distance_i_j,0,1)
                routing_path.append(node_location)
            self.cooperative_energy_consumption += energy
        return routing_path

    def custom_cooperative_routing(self):
        Q_table = self.Q_table.copy()
        routing_path = [self.env.source_node]
        while self.env.destination_node not in routing_path:

            prev_node = routing_path[-1]
            Q_table[:,prev_node] = -1000
            node_loc_index = np.where(Q_table[prev_node]==max(Q_table[prev_node]))
            node_location = self.filter_route_node(node_loc_index,prev_node)
            distance_i_j = self.env.euclidean_distance(self.env.node_locations[prev_node],self.env.node_locations[node_location])
            with open('log_file.txt', 'a') as f:
                with redirect_stdout(f):
                    print('---------Custom DEBUG starts here------------')
                    print('prev = {}, curr = {}, dist = {}'.format(prev_node,node_location,distance_i_j))
            if (distance_i_j <= self.env.r1):
                energy = cooperative_energy_consumption(distance_i_j,0,0)
                routing_path.append(node_location)
            elif (distance_i_j > self.env.r1) and (distance_i_j <= self.env.r2):
                energy_prev = cooperative_energy_consumption(distance_i_j,0,0)
                nodes_between = self.find_all_nodes_between(prev_node,node_location)
                if len(nodes_between) >= 1:
                    with open('log_file.txt', 'a') as f:
                        with redirect_stdout(f):
                            print('\n\tNODES ARE THERE IN BETWEEN FOR COOPERATION\n')
                            print("\tAvailable Nodes are: {}".format(nodes_between))
                    min_E = (2**63)-1
                    min_index = 0
                    for i in range(len(nodes_between)):
                        curr_node = nodes_between[i]
                        distance = self.env.euclidean_distance(self.env.node_locations[prev_node],self.env.node_locations[curr_node])
                        energy = attenuation(distance)
                        with open('log_file.txt', 'a') as f:
                            with redirect_stdout(f):
                                print('\tprev = {}, curr = {}, dist = {}, energy = {}'.format(prev_node,curr_node,distance,energy))
                        if energy <= min_E:
                            min_E = energy
                            min_index = i
                    if min_E < energy_prev:
                        distance_c_j = self.env.euclidean_distance(self.env.node_locations[nodes_between[min_index]],self.env.node_locations[node_location])
                        delta = 1 if distance_i_j > self.env.r1 else 0
                        energy = cooperative_energy_consumption(distance_i_j,distance_c_j,delta)
                        routing_path.append(nodes_between[min_index])
                        with open('log_file.txt', 'a') as f:
                            with redirect_stdout(f):
                                print('\tNode Taken for cooperative: {}, energy = {}'.format(nodes_between[min_index],min_E))
                    else:
                        energy = energy_prev
                        routing_path.append(node_location)
                        with open('log_file.txt', 'a') as f:
                            with redirect_stdout(f):
                                print('\n\t Cooperative node not taken due to high energy consumption. Node taken = {}, energy = {}'.format(node_location,energy))
                                print()
                else:
                    energy = cooperative_energy_consumption(distance_i_j,0,1)
                    routing_path.append(node_location)

            else:
                energy = cooperative_energy_consumption(distance_i_j,0,1)
                routing_path.append(node_location)
            self.custom_cooperative_energy_consumption += energy
        return routing_path

    def find_all_nodes_between(self,node1,node2):
        node_axis = []
        node_index = []
        for i in range(len(self.env.node_locations)):
            if i != node1 and i != node2:
                if (self.env.node_locations[i][0] > self.env.node_locations[node1][0]) and (self.env.node_locations[i][0] < self.env.node_locations[node2][0]):
                    node_index.append(i)

        return node_index

    def filter_route_node(self,node_location,last_node):
        if len(node_location) == 1:

            distance = self.env.euclidean_distance(self.env.node_locations[node_location[0][0]],self.env.node_locations[last_node])
            energy = attenuation(distance)
            index = 0
        elif len(node_location) > 1:
            min_energy = sys.maxint
            min_index = 0
            for i in range(len(node_location)):
                if self.Q_table[i][node_location] != -1000:
                    distance = self.env.euclidean_distance(self.env.node_locations[node_location[i][0]],
                                                           self.env.node_locations[last_node])
                    energy = attenuation(distance)
                    if energy <= min_energy:
                        min_energy = energy
                        min_index = i
            energy = min_energy
            index = min_index
        return node_location[index][0]

    def plot_path(self):
        non_cooperative_x = []
        non_cooperative_y = []
        cooperative_x = []
        cooperative_y = []
        custom_cooperative_x = []
        custom_cooperative_y = []
        for i in range(len(self.non_cooperative_routing_path)):
            x_l,y_l = self.env.node_locations[self.non_cooperative_routing_path[i]]
            non_cooperative_x.append(x_l)
            non_cooperative_y.append(y_l)

        for i in range(len(self.cooperative_routing_path)):
            x_l,y_l = self.env.node_locations[self.cooperative_routing_path[i]]
            cooperative_x.append(x_l)
            cooperative_y.append(y_l)

        for i in range(len(self.custom_cooperative_routing_path)):
            x_l,y_l = self.env.node_locations[self.custom_cooperative_routing_path[i]]
            custom_cooperative_x.append(x_l)
            custom_cooperative_y.append(y_l)

        plt.figure(figsize=(10,5))
        for i in range(len(self.env.node_locations)):
            plt.scatter(self.env.node_locations[i][0],self.env.node_locations[i][1],edgecolor='b',color='none')
            plt.text(self.env.node_locations[i][0],self.env.node_locations[i][1],str(i),fontsize=12)
        plt.plot(non_cooperative_x,non_cooperative_y,linestyle='-',marker='.',c='b',label='non-cooperative')
        plt.plot(cooperative_x,cooperative_y,linestyle='--',marker='.',c='r',label='cooperative')
        plt.plot(custom_cooperative_x,custom_cooperative_y,linestyle='--',marker='*',c='g',label='custom cooperative')

        plt.scatter([self.env.node_locations[self.env.source_node][0]],[self.env.node_locations[self.env.source_node][1]],c='r')
        plt.scatter([self.env.node_locations[self.env.destination_node][0]],[self.env.node_locations[self.env.destination_node][1]],c='r')
        plt.xlabel('Km')
        plt.ylabel('Km')
        plt.legend()
        plt.grid()
        plt.show()