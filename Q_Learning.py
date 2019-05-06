# Python 3.7

from QLearning.QLearning import *
import gym
import numpy as np

######################
# CREATE ENVIRONMENT #
######################

# env = gym.make('FrozenLake-v0')
env = gym.make('FrozenLake8x8-v0')


##############
# PARAMETERS #
##############

# environment settings
param_n_states = env.observation_space.n
param_n_actions = env.action_space.n
param_action_mask = ['<', '\u2193', '>', '^']

# learning settings
param_n_learn_epochs = 1000
param_n_learn_max_steps_per_epoch = 100
param_learn_rate = 0.05
param_discount_factor = 0.85

# testing settings
param_n_test_epochs = 5000
param_n_test_max_steps_per_epoch = 100


################
# CREATE AGENT #
################

qTable = QTable(param_n_states,
                param_n_actions,
                learn_rate=param_learn_rate,
                discount_factor=param_discount_factor,
                action_mask=param_action_mask)

#########
# LEARN #
#########

for i in range(param_n_learn_epochs):
    observation = env.reset()
    for _ in range(param_n_learn_max_steps_per_epoch):
        # select optimal action
        act = qTable.optimal_action(observation)

        # choose to explore or exploit
        act = discrete_action_generator(act, param_n_actions, 1 - (i / param_n_learn_epochs))

        # take action
        new_observation, reward, done, info = env.step(act)

        # define reward
        if done and reward == 0.0:
            reward = -0.5
        elif not done:
            reward = -0.01

        # update Q_table
        qTable.learn(observation, new_observation, act, reward)

        # update last state
        observation = new_observation

        # quite when goal is reached
        if done:
            break


########
# TEST #
########

total_reward = 0
runOut = []
for loop in range(param_n_test_epochs):
    observation = env.reset()
    for step in range(param_n_test_max_steps_per_epoch):
        act = qTable.optimal_action(observation)
        observation, reward, done, info = env.step(act)

        if done:
            total_reward += reward
            runOut.append(step)
            break


################
# PRINT RESULT #
################

env.render()
print()
print("Total Reward: {:0.2f}%".format((total_reward / param_n_test_epochs) * 100))
print("Run out: {:0.0f}".format(np.mean(runOut)))
print()
print("Q action map")
qTable.print_action_table_masked(8)
print()
print("Action Table")
qTable.print_action_table(8)
print()
print("Value Table")
qTable.print_value_table(8)
print()
print("Q Table")
qTable.print_q_table()
