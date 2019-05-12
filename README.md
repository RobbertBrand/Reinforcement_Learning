#Reinforcement Learning
A Q Learning implementation in python, learning to play an OpenAI Gym game.

## Description
This script creates several agents. Each agent learns to play the same OpenAI Gym FrozenLake-v0 game, using QTables.
  
The FrozenLake-v0 game, consists out of a 4 by 4 board. Each position on the board is described by a letter:
 * S Starting point
 * F Walkable frozen surface
 * H Hole, ending the game by failure
 * G Goal, ending the game by victory
  
The charter in the game, start at the position marked by S. The goal is to move over the frozen position, to reach the
goal. The game is ended by reaching the goal which results in a reward, or by falling in a hole resulting in failure.
  
```
SFFF
FHFH
FFFH
HFFG
```
  
The frozen position are slippery. Because of this, might the charter move in a different position than requested by the
agent. This makes the game tricky and impossible to succeed 100% of the time, playing the game.
  
## Installation
Install Python 3.7 with the following Python packages in your environment:
 * numpy v1.16.2
 * gym v0.12.0
  
You are ready to roll!

## Usage
Run the ReinforcementLearning.py program to train and test agents.
```
>> python ReinforcementLearning.py
```

###Result
Each trained and tested agent returns it's learned policy and value table.
```
Q learning cycle 10

Q action map
|  <   ^   <   ^  |
|  <       <      |
|  ^   ↓   <      |
|      >   ↓      |

Value Table
|  0.5124  0.4670  0.3997  0.2609  |
|  0.5252  0.0000  0.2320  0.0000  |
|  0.5431  0.5853  0.5660  0.0000  |
|  0.0000  0.7097  0.8437  0.0000  |
```
  
The scores displayed after training and testing all agents.
```
Mean Reward over cycles: 75.40%
Reward per cycle:        81.00% 71.00% 75.00% 79.00% 76.00% 70.00% 74.00% 78.00% 72.00% 78.00%
Mean steps over cycles:  39
Steps per cycle:         43     39     36     38     40     39     38     36     42     39     
```

## Discussion
At first sight seems the policy, learned by the agents, a bit odd.
```
Q action map
|  <   ^   <   ^  |
|  <       <      |
|  ^   ↓   <      |
|      >   ↓      |
```
The policy seems to send the charter straight in to a hole or the game border. This for example at the start position
or at the position 3 right 1 down from the start.
  
This odd phenomenon might be explainable by the theory that the chance that the charter moves in the commanded
direction by the agent, is smaller than the chance that the charter moves left or right relative to the agents command
direction. This because of the slipperiness of the frozen positions.

## Result
The best observed result by a single agent, playing 100 games, is 81%. The best observed average result over 10 agents,
all trained using the same settings and each playing 100 games, is 75.40%
  
Results might be further improved, by tweaking parameters and adjusting the reward system.  

## Reference
[Link](https://gym.openai.com/envs/FrozenLake-v0/) OpenAI Gym, Frozen Lake game 