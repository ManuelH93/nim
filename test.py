import math
import random
import time

class NimAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is a tuple of remaining piles, e.g. (1, 1, 4, 4)
         - `action` is a tuple `(i, j)` for an action
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        key = (tuple(state), action)
        q_value = self.q.get(key,0)
        return q_value

ai = NimAI()

ai.q[(1, 1, 4, 4), (0, 2)] = 10100

state = [1, 1, 4, 4]
action = (0,2)

print(ai.get_q_value(state, action))
#print(ai.q[(1, 1, 4, 4), (0, 2)])


