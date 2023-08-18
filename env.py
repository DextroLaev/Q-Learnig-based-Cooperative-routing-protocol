import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self,nnodes,n_states,n_actions,x_range,y_range):
        self.nnodes = nnodes
        self.n_states = n_states
        self.n_actions = n_actions
        self.r1 = 2.5
        self.r2 = 4
        self.x_range = x_range
        self.y_range = y_range
        self.node_locations = self.init_node_locations()

        self.source_node,self.destination_node = self.select_source_destination_node()
        self.edge,self.reward_table = self.design_reward_table()
        self.state = None

    # Reward table is filled with the euclidean distance of the nodes.
    def design_reward_table(self):
        edge = np.zeros((self.nnodes,self.nnodes))  #contains the distance of the nodes.
        # The reward table below contains the negative distance of every nodes with each other
        reward_table = np.zeros((self.n_states,self.n_actions))
        for i in range(len(self.node_locations)):
            for j in range(len(self.node_locations)):
                edge[i][j] = self.euclidean_distance(self.node_locations[i],self.node_locations[j])

                # cooperative routing algo
                if i == j:

                    # Reward of K to K nodes is set as -1000 for error controlling.
                    reward_table[i][j] = -1000
                else:
                    reward_table[i][j] = -edge[i][j]
        return edge,reward_table

    # Randomly nodes are distributed under the sea
    def init_node_locations(self):
        x_range = self.x_range
        y_range = self.y_range
        x_nodes_loc = np.linspace(0,x_range,self.nnodes)
        y_nodes_loc = np.linspace(0,y_range,self.nnodes)
        np.random.shuffle(x_nodes_loc)
        np.random.shuffle(y_nodes_loc)
        node_loc = []
        for i in range(self.nnodes):
            node_loc.append((x_nodes_loc[i],y_nodes_loc[i]))
        return node_loc

    # all the nodes are plotted on the graph
    def plot_node_locations(self):
        for i in range(len(self.node_locations)):
            if i == self.source_node or i == self.destination_node:
                c = 'r'
            else:
                c = 'b'
            plt.scatter(self.node_locations[i][0],self.node_locations[i][1],c=c)
        plt.show()

    def euclidean_distance(self,node1,node2):
        first_part = np.square(node2[0] - node1[0])
        second_part = np.square(node2[1] - node1[1])
        return np.sqrt(first_part+second_part)

    # Step function returns the Next_state,reward for the state and done=Either training is done or not.
    def step(self,action):
        done = False
        reward = self.reward_table[self.state][action]
        if self.destination_node == action:
            done = True
        if reward > -1000:
            self.state = action
        else:
            done = True
        return self.state,reward,done

    # The min and max distance are selected as source node and destination node.
    def select_source_destination_node(self):
        min_node_index = self.node_locations.index(min(self.node_locations))
        max_node_index = self.node_locations.index(max(self.node_locations))
        return min_node_index,max_node_index