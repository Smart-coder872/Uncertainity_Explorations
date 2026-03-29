from gridworld import GridWorld
from reward import Reward
from transition import Transition
import numpy as np
import matplotlib.pyplot as plt

def initialize_value(grid_world):
    """Given a grid world, initialize the value function."""
    Vhat = np.zeros(len(grid_world.grid))
    return Vhat

def value_iteration(grid_world, actions, discount, epsilon):
    """Perform value iteration.
    Inputs:
        grid_world: a GridWorld class
        actions: a list of possible actions a robot can take
        discount: the discount factor to apply to rewards
        episilon: the convergence threshold
    Outputs:
        Vhat: the converged value function over all world states
    """
    Vhat = initialize_value(grid_world)
    while True:
        delta = 0
        for state in grid_world.grid.keys():
            v = Vhat[state]  # store the current value for checking convergence
            Vhat[state] = max(sum(grid_world.compute_action_probability(grid_world.grid[state], action, grid_world.grid[next_state]) * 
                                (grid_world.compute_reward(grid_world.grid[state], action, grid_world.grid[next_state]) + discount * Vhat[next_state]) 
                                for next_state in grid_world.grid.keys()) for action in actions)
            delta = max(delta, abs(v - Vhat[state]))
        if delta < epsilon:
            break
    return Vhat

def policy_selection(grid_world, Vhat, discount):
    """Compute the optimal policy.
    Inputs:
        grid_world: A GridWorld class
        Vhat: a converged value distribution from value iteration
        discount: the discount factor applied to rewards
    Outputs:
        policy: a lookup table (dictionary) of best actions for every state in grid_world
    """
    policy = {}
    for state in grid_world.grid.keys():
        policy[state] = max(actions, key=lambda a: sum(grid_world.compute_action_probability(grid_world.grid[state], a, grid_world.grid[next_state]) * 
                                                       (grid_world.compute_reward(grid_world.grid[state], a, grid_world.grid[next_state]) + discount * Vhat[next_state])
                                                       for next_state in grid_world.grid.keys()))
    return policy

if __name__ == "__main__":
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

    # Value Iteration and Optimal Policy Computation
    Vhat = value_iteration(world, actions, discount, convergence_threshold)
    policy = policy_selection(world, Vhat, discount)

    # Plotting
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
    plot_actions = {"stay":(0,0), "right":(0.4,0), "left":(-0.4,0), "up":(0,0.4), "down":(0,-0.4)}
    ax[0].imshow(Vhat.reshape(xdim, ydim).T, origin="lower")
    ax[0].set_title(f"Value Iteration -- discount:{discount}, payoff:{payoff}, goal:{payoff_loc}")

    ax[1].imshow(Vhat.reshape(xdim, ydim).T, origin="lower")
    for key, value in policy.items():
        xval = world.grid[key][0]+plot_actions[value][0]
        yval = world.grid[key][1]+plot_actions[value][1]
        ax[1].arrow(world.grid[key][0], world.grid[key][1], plot_actions[value][0], plot_actions[value][1], width=0.1)
    ax[1].set_title("Optimal Policy")
    plt.show()