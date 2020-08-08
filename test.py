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

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estiamte of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        new_q = reward + future_rewards
        self.q[tuple(state), action] = old_q + self.alpha * (new_q - old_q)


ai = NimAI()

ai.q[(1, 1, 4, 4), (0, 2)] = 0.75

state = [1, 1, 4, 4]
action = (0,2)

old_q = ai.get_q_value(state,action)
reward = 0
future_rewards = 1

ai.update_q_value(state, action, old_q, reward, future_rewards)

print(ai.q)


