import gym
import random
import time
import itertools
import sys
from pprint import pprint
from pong_NeuralNetwork2 import neuralNetwork

game = 'Pong-ram-v0'
env = gym.make(game)
print(str(env.observation_space.shape[0]) + ' ' + str(env.action_space.n))

new = False

for i, x in enumerate(sys.argv):
	if x == '--model':
		model_number = sys.argv[i+1]
	if x == '--new':
		new = True

nn = neuralNetwork(game=game, observations=(env.observation_space.shape[0]*2), actions=env.action_space.n, model_number=model_number, new=new)

high_score = -30
average = 0
games_per_render = nn.configurations['episodePerTrain']
for episode in range(1000):
	observation_list = [[0]*env.observation_space.shape[0]]
	observation = env.reset()
	observation_list.append(observation)
	observation = list(itertools.chain.from_iterable(observation_list))
	points = 0
	i = 1
	while True:
		if (i%4 == 0):
			# if nn.configurations['episodes'] % games_per_render == 0 and nn.configurations['episodes'] > 100:
			# 	env.render()
			# 	time.sleep(.01)

			old_observation = observation

			action = nn.prediction(observation)
			observation, reward, done, info = env.step(action)

			points += reward

			observation_list.pop(0)
			observation_list.append(observation)
			observation = list(itertools.chain.from_iterable(observation_list))

			if done:
				# if reward != 0 and nn.configurations['episodes'] % games_per_render == 0:
				# 	print(reward)
				if points > high_score:
					high_score = points
				nn.model_memory.append([old_observation, action, reward, observation])
				break

			# if reward != 0 and nn.configurations['episodes'] % games_per_render == 0:
			# 	print(reward)
				
			nn.model_memory.append([old_observation, action, reward, observation])
		i+=1


	print("\n\nGame: "+str(nn.configurations['episodes'])+"\n\n\n")
	print("Score this game: " + str(points))
	print("Highest score:   " + str(high_score))
	print("Epsilon:         " + str(nn.configurations['epsilon']) + "\n\n")




	nn.replay()

	print()