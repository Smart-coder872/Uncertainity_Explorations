from gridworld import GridWorld
from reward import Reward
from transition import Transition
import numpy as np
import matplotlib.pyplot as plt
import copy

def initialize_value(grid_world):
    """Given a grid world, initialize the value function."""
    Vhat = np.zeros(len(grid_world.grid))
    return Vhat

def initialize_policy(grid_world, actions):
    """Given a grid world and set of actions, initialize the policy."""
    pihat = dict()
    for key in grid_world.grid.keys():
        pihat[key] = np.random.choice(actions)
    return pihat

def policy_iteration(grid_world, actions, discount):
    """Performs policy iteration for a given Gridworld."""
    Vhat = initialize_value(grid_world)
    pihat = initialize_policy(grid_world, actions)

    while True:
        p = copy.deepcopy(pihat)  # current policy; will compare after refinement
        for state in grid_world.grid.keys():  # update the value function
            Vhat[state] = sum(grid_world.compute_action_probability(grid_world.grid[state], pihat[state], grid_world.grid[next_state]) * 
                              (grid_world.compute_reward(grid_world.grid[state], pihat[state], grid_world.grid[next_state]) + discount * Vhat[next_state]) 
                              for next_state in grid_world.grid.keys())
        for state in grid_world.grid.keys():  # refine the policy
            pihat[state] = max(actions, key=lambda a: sum(grid_world.compute_action_probability(grid_world.grid[state], a, grid_world.grid[next_state]) * 
                                                       (grid_world.compute_reward(grid_world.grid[state], a, grid_world.grid[next_state]) + discount * Vhat[next_state])
                                                       for next_state in grid_world.grid.keys()))
            
        if p == pihat:  # policy has converged
            break

    return Vhat, pihat

if __name__ == "__main__":
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
   
    Vhat, policy = policy_iteration(world, actions, discount)