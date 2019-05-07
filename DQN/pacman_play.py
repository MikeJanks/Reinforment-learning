import gym
import random
import time
import itertools
import sys
import numpy
import tensorflow as tf
from pprint import pprint
from pacman_NeuralNetwork import neuralNetwork

game = 'MsPacman-ram-v0'
env = gym.make(game)
print('Enviroment: ' + str(env.observation_space.shape[0]) + '	Actions:' + str(env.action_space.n))
# print(env.unwrapped.get_action_meanings())
nn = tf.keras.models.load_model('.\\NeuralNetwork_saved_data\\pacman_'+str(sys.argv[1])+'_model.nn')
high_score = 0
average = 0
for episode in range(1000):
	print("\n\nGame: "+str(episode+1)+"\n\n")
	observation_list = [[0]*env.observation_space.shape[0],[0]*env.observation_space.shape[0]]
	observation = env.reset()
	observation_list.append(observation)
	observation = list(itertools.chain.from_iterable(observation_list))
	lives = 3
	points = 0
	i = 0
	while True:
		if i%2 == 0:
			env.render()
			time.sleep(.025)
	
			actions = nn.predict(numpy.array([observation]))[0]
			action = numpy.argmax(actions)

			# print(actions)
			# input()

			observation, reward, done, info = env.step(action)

			# pprint(info)

			points += reward

			observation_list.pop(0)
			observation_list.append(observation)
			# pprint(observation_list)
			observation = list(itertools.chain.from_iterable(observation_list))


			if done:
				reward -= 1000
				# if reward != 0:
				# 	print(reward)
				if points > high_score:
					high_score = points
				break

			# if reward != 0:
			# 	print(reward)
		i+=1

	print("\n\nScore this game: " + str(points))
	print("\nHighest score:   " + str(high_score) + "\n\n")

	print()
