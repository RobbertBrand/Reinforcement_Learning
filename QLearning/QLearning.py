"""
    Q Learning basic toolset, Made by Robbert Brand
"""

import numpy as np


def q_learn(old_state_action_q_value, new_state_max_q_value, reward, learn_rate=0.01, discount_factor=0.9):
    """
    Returns updated state-action Q_value.
    :param old_state_action_q_value: Q_value of the chosen action in the previous state.
    :param new_state_max_q_value: maximum Q_value of new state.
    :param reward: received reward as result of taken action.
    :param learn_rate: learning rate.
    :param discount_factor: discount factor.
    :return: Updated Q-value of previous state-action pair.
    """
    error = (reward + (discount_factor * new_state_max_q_value)) - old_state_action_q_value
    return old_state_action_q_value + (learn_rate * error)


def discrete_action_generator(preferred_action, action_space, exploration_factor):
    """
    Returns preferred or random action, based on exploration factor. Actions are of the integer type starting from 0.
    :param preferred_action: desired action.
    :param action_space: amount of available actions.
    :param exploration_factor: chance to pick preferred action.
    :return: selected action. Could be the preferred action or random action, based on chance.
    """
    random_action = np.random.choice(action_space)
    return np.random.choice([preferred_action, random_action], p=[1 - exploration_factor, exploration_factor])
