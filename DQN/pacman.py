import gym
import random
import time
import itertools
import sys
from pprint import pprint
from pacman_NeuralNetwork import neuralNetwork

game = 'MsPacman-ram-v0'
env = gym.make(game)
# print(str(env.observation_space.shape[0]) + ' ' + str(env.action_space.n))

new = False

for i, x in enumerate(sys.argv):
	if x == '--model':
		model_number = sys.argv[i+1]
	if x == '--new':
		new = True

nn = neuralNetwork(observations=(env.observation_space.shape[0]*3), actions=env.action_space.n, model_number=model_number, new=new)

high_score = 0
average = 0
games_per_render = nn.episodePerTrain
for episode in range(1000):
	observation_list = [[0]*env.observation_space.shape[0],[0]*env.observation_space.shape[0]]
	observation = env.reset()
	observation_list.append(observation)
	observation = list(itertools.chain.from_iterable(observation_list))
	lives = 3
	points = 0
	i = 1
	while True:
		if (i%2 == 0):
			if nn.model_memory['episodes'] % games_per_render == 0 and nn.model_memory['episodes'] > 5:
				env.render()
				time.sleep(.01)
			# env.render()
			# time.sleep(.01)

			nn.model_memory['memoryState'].append(observation)

			action = nn.prediction(observation)

			nn.model_memory['memoryAction'].append(action)
			observation, reward, done, info = env.step(action)

			points += reward

			observation_list.pop(0)
			observation_list.append(observation)
			# pprint(observation_list)
			observation = list(itertools.chain.from_iterable(observation_list))

			nn.model_memory['memoryNextState'].append(observation)

			if done:
				reward -= 1000
				# if reward != 0 and nn.model_memory['episodes'] % games_per_render == 0:
				# 	print(reward)
				if points > high_score:
					high_score = points
				nn.model_memory['memoryReward'].append(reward)
				break

			if info["ale.lives"] < lives:
				reward -= 500
				lives = info["ale.lives"]

			# if reward != 0 and nn.model_memory['episodes'] % games_per_render == 0:
			# 	print(reward)
			nn.model_memory['memoryReward'].append(reward)
		i+=1


	print("\n\nGame: "+str(nn.model_memory['episodes'])+"\n\n\n")
	print("Score this game: " + str(points))
	print("Highest score:   " + str(high_score))
	print("Epsilon:         " + str(nn.epsilon) + "\n\n")




	nn.replay()

	print()
