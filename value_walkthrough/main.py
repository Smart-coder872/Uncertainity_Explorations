from gridworld import GridWorld
from reward import Reward
from transition import Transition
import numpy as np
import matplotlib.pyplot as plt
import copy

    # Problem Set-Up
actions = ["stay","right","left","up","down"]
xdim = 5
ydim = 5
discount = 0.9
convergence_threshold = 0.0001
payoff = 100
payoff_loc = (1,3)
cost = -1
reward = Reward(payoff_loc, payoff=payoff, cost=cost)
transition = Transition()
world = GridWorld(xdim, ydim, reward, transition)

iteration_type = 0

if iteration_type == 0:
    import v_iteration
elif iteration_type == 1:
    import policy_iteration
else:
    print("iteration type not recognized")