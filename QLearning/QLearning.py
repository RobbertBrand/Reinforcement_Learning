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


class QTable:
    """
    Q Table implementation.

    PSEUDO CODE EXAMPLE:
        # LEARN
        for i in range(n_epochs):
            observation = environment.observe()
            for _ in range(max_steps_per_epoch):
                # select optimal action
                act = qTable.optimal_action(observation)

                # choose to explore or exploit
                act = discrete_action_generator(act, action_space, 1 - (i / n_epochs))

                # take action
                new_observation, reward, done = environment.take_action(act)

                # update Q_table
                qTable.learn(observation, new_observation, act, reward)

                # update last state
                observation = new_observation

                # quite when goal is reached
                if done:
                    break

        # PLAY
        observation = environment.observe()
        for step in range(param_n_test_max_steps_per_epoch):
            act = qTable.optimal_action(observation)
            observation, reward, done, info = env.step(act)

            if done:
                break
    """

    def __init__(self, n_states, n_actions, learn_rate=0.01, discount_factor=0.9, action_mask=None):
        """
        Initialize Q Table
        :param n_states: amount of environment states.
        :param n_actions: amount of possible actions.
        :param learn_rate: learning rate.
        :param discount_factor: discount factor.
        :param action_mask: translation from actions, being integer type, to charters. A blank charter will be added
                            if mask length equals action space length. Extra is meant for -1 action, in case of invalid
                            states.
                            Example value, for action space 4:
                                ['<', '\u2193', '>', '^']
        """
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
        """
        returns q_value's for requested state.
        :param state: state.
        :return: list of Q_value's.
        """
        return self.qTable[state][1]

    def state_status(self, state):
        """
        returns the status of a requested state. State status is -1, if q_value's where not updated at least once.
        :param state: state.
        :return: state status.
        """
        return self.qTable[state][0]

    def set_state_valid(self, state, valid=0):
        """
        set state status.
        :param state: state.
        :param valid: status.
        """
        self.qTable[state][0] = valid

    # INDIRECT Q TABLE OPS
    ######################
    def is_state_valid(self, state):
        """
        returns state validity. True when valid. State will become valid after at least one q_value in this state has
        bin updated.
        :param state: state.
        :return: True if state is valid.
        """
        return self.state_status(state) >= 0

    def optimal_action(self, state):
        """
        returns optimal action for requested state, based on q_value's.
        :param state: state.
        :return: action as integer.
        """
        return np.argmax(self[state])

    def optimal_action_valid(self, state, negative_when_invalid=True):
        """
        returns optimal action for requested state, based on q_value's. Action will be -1, if state is invalid. State
        will become valid, after at least one q_value was updated.
        :param state: state.
        :param negative_when_invalid: function will return -1 when state is invalid and this parameter is true.
                                      function will return action, even when state is invalid, in case this parameter
                                      is false.
        :return: action
        """
        action = -1
        if (not negative_when_invalid) or self.is_state_valid(state):
            action = self.optimal_action(state)
        return action

    def optimal_action_masked(self, state, hide_action_for_invalid_state=False):
        """
        returns optimal action from action mask for requested state, based on q_value's.
        :param state: state.
        :param hide_action_for_invalid_state: hide action gor invalid state, when true.
        :return: masked optimal action.
        """
        return self.action_mask[self.optimal_action_valid(state, hide_action_for_invalid_state)]

    def optimal_action_value(self, state):
        """
        return highest q_value of requested state.
        :param state: state
        :return: highest q_value
        """
        return max(self[state])

    def learn(self, old_state, new_state, action, reward):
        """
        update q_value after taking an action.
        :param old_state: state before taking action.
        :param new_state: state after taking action.
        :param action: taken action.
        :param reward: received reward, as result of taken action.
        """
        self.set_state_valid(old_state)
        self[old_state][action] = q_learn(self[old_state][action],
                                          self.optimal_action_value(new_state),
                                          reward,
                                          self.learn_rate,
                                          self.discount_factor)

    # PRINTING TOOLS
    ################
    def print_action_table(self, table_row_length=1, hide_invalid_action=True):
        """
        print best q_value per state.
        :param table_row_length: print given amount states per row.
        :param hide_invalid_action: hide invalid actions when true.
        """
        printable = [self.optimal_action_valid(i, hide_invalid_action) for i in range(len(self.qTable))]
        print_table_from_flat_list(printable, table_row_length)

    def print_action_table_masked(self, table_row_length=1, hide_invalid_action=True):
        """
        print best action from mask per state.
        :param table_row_length: print given amount states per row.
        :param hide_invalid_action: hide invalid actions when true.
        """
        assert self.action_mask is not None
        printable = [self.action_mask[self.optimal_action_valid(i, hide_invalid_action)] for i in
                     range(len(self.qTable))]
        print_table_from_flat_list(printable, table_row_length)

    def print_value_table(self, table_row_length=1):
        """
        print best q_value per state.
        :param table_row_length: print given amount states per row.
        """
        printable_q_table = [[self.optimal_action_value(i)] for i in range(len(self.qTable))]
        print_table_from_flat_list(printable_q_table, table_row_length)

    def print_q_table(self, table_row_length=1):
        """
        print q_table.
        :param table_row_length: print given amount states per row.
        :return:
        """
        printable_q_table = [self[i] for i in range(len(self.qTable))]
        print_table_from_flat_list(printable_q_table, table_row_length)


# TODO: Move to Basic lib
def print_table_from_flat_list(flat_list, table_row_length):
    """
    print table from a flat list. flat list is divided in rows, based on given parameters.
    :param flat_list: input list.
    :param table_row_length: print given amount flat_list items per row.
    """
    print_format = print_format_creator(flat_list[0], table_row_length)
    for row in range(0, len(flat_list), table_row_length):
        print('|', print_format.format(*flat_list[row:row + table_row_length]), '|')


# TODO: Move to Basic lib
def print_format_creator(type_spec, length):
    """
    returns string format spec, to print a list of items. Each item in the list should be of the same type.
    :param type_spec: data sample, to determine type from.
    :param length: amount of items to be printed.
    :return: format string.
    """
    if type(type_spec) is int:
        print_format = ''.join(['{:5.0}'] * length)
    elif type(type_spec) is float:
        print_format = ''.join([' {:0.4f} '] * length)
    else:
        print_format = ' '.join([' {} '] * length)
    return print_format
