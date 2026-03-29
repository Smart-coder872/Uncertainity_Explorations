class Transition(object):
    '''Create a transition function object for a gridworld.'''
    def __init__(self):
        pass

    def compute_transition(self, state, action, next_state):
        '''For a given state and action, compute the probability of the next state.'''
        if action == "stay" and state == next_state:
            return 1.0
        elif action == "right":
            if next_state[0] == state[0] + 1 and next_state[1] == state[1]:
                return 0.8
            elif next_state[0] == state[0] - 1 and next_state[1] == state[1]:
                return 0.2
            else:
                return 0.0
        elif action == "left":
            if next_state[0] == state[0] - 1 and next_state[1] == state[1]:
                return 0.8
            elif next_state[0] == state[0] + 1 and next_state[1] == state[1]:
                return 0.2
            else:
                return 0.0
        elif action == "up":
            if next_state[1] == state[1] + 1 and next_state[0] == state[0]:
                return 0.8
            elif next_state[1] == state[1] - 1 and next_state[0] == state[0]:
                return 0.2
            else:
                return 0.0
        elif action == "down":
            if next_state[1] == state[1]-1 and next_state[0] == state[0]:
                return 0.8
            elif next_state[1] == state[1] + 1 and next_state[0] == state[0]:
                return 0.2
            else:
                return 0.0
        else:
            return 0.0