from reward import Reward
from transition import Transition

class GridWorld(object):
    def __init__(self, xdim, ydim, reward, transition):
        self.xdim = xdim  # number of columns in the world
        self.ydim = ydim  # number of rows in the world
        self.reward = reward  # reward function
        self.transition = transition  # transition function

        self._make_grid()  # populate the world
    
    def _make_grid(self):
        """Create the states for the grid world"""
        self.grid = dict()
        grid = []
        for i in range(self.xdim):
            for j in range(self.ydim):
                grid.append((i, j))
        for i, coord in enumerate(grid):
            self.grid[i] = coord

    def compute_action_probability(self, state, action, next_state):
        """Convenience function for computing transition probability."""
        return self.transition.compute_transition(state, action, next_state)
    
    def compute_reward(self, state, action, next_state):
        """Convenience function for computing immediate reward."""
        return self.reward.compute_reward(state, action, next_state)