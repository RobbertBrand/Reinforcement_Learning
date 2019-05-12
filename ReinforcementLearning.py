"""
    Reinforcement Learning implementation, Made by Robbert Brand.

    Runs on Python 3.7
"""

from QLearning.QTable.QTable import *
import gym
import numpy as np

######################
# CREATE ENVIRONMENT #
######################

env = gym.make('FrozenLake-v0')
# env = gym.make('FrozenLake8x8-v0')

##############
# PARAMETERS #
##############

# environment settings
param_n_states = env.observation_space.n
param_n_actions = env.action_space.n
#                   ['LEFT',   'DOWN',   'RIGHT',  'UP',     'NON']
# param_action_mask = ['\u2b9C', '\u2b9f', '\u2b9e', '\u2b9d', ' ']
param_action_mask = ['<', '\u2193', '>', '^']

# learning settings
param_n_learn_epochs = 4000
param_n_learn_max_steps_per_epoch = 100
param_train_test_cycles = 10

param_learn_rate = 0.05
param_discount_factor = 0.99

# reward settings
param_reward_step = 0.0
param_reward_lost = 0.0
param_reward_win = 1.0

# testing settings
param_n_test_epochs = 100
param_n_test_max_steps_per_epoch = 100

#########################
# RUN TRAIN TEST CYCLES #
#########################

train_test_cycle_reward_list = []
train_test_cycle_runout_list = []

for j in range(param_train_test_cycles):
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
                reward = param_reward_lost
            elif done and reward == 1.0:
                reward = param_reward_win
            elif not done:
                reward = param_reward_step

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

    train_test_cycle_reward_list.append((total_reward / param_n_test_epochs) * 100)
    train_test_cycle_runout_list.append(np.mean(runOut))

    ######################
    # PRINT CYCLE RESULT #
    ######################

    print("Q learning cycle {}".format(j))
    print()
    print("Q action map")
    qTable.print_action_table_masked(np.sqrt(param_n_states).astype(int))
    print()
    # print("Action Table")
    # qTable.print_action_table(np.sqrt(param_n_states).astype(int))
    # print()
    print("Value Table")
    qTable.print_value_table(np.sqrt(param_n_states).astype(int))
    print()
    # print("Q Table")
    # qTable.print_q_table(np.sqrt(param_n_states).astype(int))
    # print()
    print()


################
# PRINT RESULT #
################

print("Train Test cycles result")
print()
env.render()
print()
print("Mean Reward over cycles: {:0.2f}%".format(np.mean(train_test_cycle_reward_list)))
print(("Reward per cycle:       " + (" {:0.2f}%" * param_train_test_cycles)).format(*train_test_cycle_reward_list))
print("Mean steps over cycles:  {:0.0f}".format(np.mean(runOut)))
print(("Steps per cycle:        " + (" {:0.0f}    " * param_train_test_cycles)).format(*train_test_cycle_runout_list))
print()
