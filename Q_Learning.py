# Python 3.7

import gym
import numpy

# env = gym.make('FrozenLake-v0')
env = gym.make('FrozenLake8x8-v0')
print(env.observation_space)

qTable = [[0.0, 0.0, 0.0, 0.0] for i in range(64)]

print(env.action_space)
print(env.observation_space)
for i in range(10000):
    observation = env.reset()
    for _ in range(100):
        # print()
        # print()

        # env.render()

        logic_act = numpy.argmax(qTable[observation])
        random_act = numpy.random.choice(4)
        act = numpy.random.choice([logic_act, random_act], p=[(i / 10000.0), 1 - (i / 10000.0)])

        new_observation, reward, done, info = env.step(act)

        if done and reward == 0.0:
            reward = -1.0
        # elif done and reward > 0.0:
        #     reward = 10
        elif not done:
            reward = -0.05

        # for row in qTable:
        #     print('{:0.4f}  {:0.4f}  {:0.4f}  {:0.4f}'.format(*row))

        # print("obs: ", observation, " > ", new_observation)
        # print("act: ", act)
        # print("reward", reward)

        qTable[observation][act] = qTable[observation][act] + (0.01 * (reward + (0.9 * max(qTable[new_observation])) - qTable[observation][act]))

        observation = new_observation
        # time.sleep(1)

        if done:
            break

total_reward = 0
runOut = []
total_runs = 1000
for loop in range(total_runs):
    observation = env.reset()
    for step in range(100):
        # env.render()

        act = numpy.argmax(qTable[observation])
        # act = numpy.random.choice(4)
        observation, reward, done, info = env.step(act)

        if done:
            total_reward += reward
            runOut.append(step)
            break

env.render()

print("Total Reward: {}%".format((total_reward / total_runs) * 100))
print("Run out: {}".format(numpy.mean(runOut)))


print()
move = ['<', '\u2193', '>', '^']

for row in range(8):
    test = []
    for col in range(8):
        if col == 7 and row == 7:
            direction = '*'
        elif max(qTable[(row * 8) + col]) == 0.0:
            direction = ' '
        else:
            direction = move[numpy.argmax(qTable[(row * 4) + col])]
        test.append(direction)
    print("| {}  {}  {}  {} {}  {}  {}  {} |".format(*test))

print()
print()

for row in range(8):
    test = []
    for col in range(8):
        test.append('{:0.4f}    {:0.4f}    {:0.4f}    {:0.4f}  |'.format(*qTable[(row * 4) + col]))
    print(test)

print()
print()
