import numpy as np


def q_learn(old_state_action_value, new_state_value, reward, learn_rate=0.01, discount_factor=0.9):
    error = (reward + (discount_factor * new_state_value)) - old_state_action_value
    return old_state_action_value + (learn_rate * error)


def discrete_action_generator(prefered_action, action_space, exploration_factor):
    random_action = np.random.choice(action_space)
    return np.random.choice([prefered_action, random_action], p=[1 - exploration_factor, exploration_factor])


class QTable:
    def __init__(self, n_states, n_actions, learn_rate=0.01, discount_factor=0.9, action_mask=None):
        assert action_mask is None or len(action_mask) == n_actions or len(action_mask) == n_actions + 1

        if action_mask is not None and len(action_mask) == n_actions:
            action_mask.append(' ')

        self.n_states = n_states
        self.n_actions = n_actions
        self.qTable = np.array([[-1, [0.0 for _ in range(n_actions)]] for _ in range(n_states)])
        self.learn_rate = learn_rate
        self.discount_factor = discount_factor
        self.action_mask = action_mask

    # DIRECT Q TABLE OPS
    ####################
    def __getitem__(self, state):
        return self.qTable[state][1]

    def state_status(self, state):
        return self.qTable[state][0]

    def set_state_valid(self, state, valid=0):
        self.qTable[state][0] = valid

    # INDIRECT Q TABLE OPS
    ######################
    def is_state_valid(self, state):
        return self.state_status(state) >= 0

    def optimal_action(self, state):
        return np.argmax(self[state])

    def optimal_action_valid(self, state, negative_when_invalid=True):
        action = -1
        if (not negative_when_invalid) or self.is_state_valid(state):
            action = self.optimal_action(state)
        return action

    def optimal_action_masked(self, state, negative_when_invalid=False):
        return self.action_mask[self.optimal_action_valid(state, negative_when_invalid)]

    def optimal_action_value(self, state):
        return max(self[state])

    def learn(self, old_state, new_state, action, reward):
        self.set_state_valid(old_state)
        self[old_state][action] = q_learn(self[old_state][action],
                                          self.optimal_action_value(new_state),
                                          reward,
                                          self.learn_rate,
                                          self.discount_factor)

    # PRINTING TOOLS
    ################
    def print_action_table(self, table_row_length=1, hide_invalid_action=True):
        printable = [self.optimal_action_valid(i, hide_invalid_action) for i in range(len(self.qTable))]
        print_table_from_flat_list(printable, table_row_length)

    def print_action_table_masked(self, table_row_length=1, hide_invalid_action=True):
        printable = [self.action_mask[self.optimal_action_valid(i, hide_invalid_action)] for i in
                     range(len(self.qTable))]
        print_table_from_flat_list(printable, table_row_length)

    def print_value_table(self, table_row_length=1):
        printable_q_table = [[self.optimal_action_value(i)] for i in range(len(self.qTable))]
        print_table_from_flat_list(printable_q_table, table_row_length)

    def print_q_table(self, table_row_length=1):
        printable_q_table = [self[i] for i in range(len(self.qTable))]
        print_table_from_flat_list(printable_q_table, table_row_length)


def print_table_from_flat_list(flat_list, table_row_length):
    for row in range(0, len(flat_list), table_row_length):
        print('|', ' '.join(flat_list[row:row + table_row_length]), '|')
