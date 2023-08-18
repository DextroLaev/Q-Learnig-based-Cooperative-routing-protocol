import agent
import env
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
	env = env.Environment(nnodes=18,n_states=18,n_actions=18,x_range=14,y_range=5)
	agent = agent.Agent(env,0.8,0.1)
	agent.run_agent(1500)
	agent.select_routing_path()
	agent.plot_path()