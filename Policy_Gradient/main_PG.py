import gym, random, time, itertools, numpy, sys
from PG import neuralNetwork
from tqdm import tqdm
import tensorflow as tf
from funtions import *


new = False
for i, x in enumerate(sys.argv):
	if x == '--model':
		model_number = sys.argv[i+1]
	if x == '--new':
		new = True


game = 'Pong-v0'
env = gym.make(game)
game_state_size = 84

nn = neuralNetwork(game=game, observations=game_state_size, actions=env.action_space.n, model_number=model_number, new=new)

all_games_states, all_games_actions, all_games_discount_rewards = [], [], []


print('Starting games...\n')
while True:

	states, actions, rewards, points = play_game(nn, render=True)

	nn.configurations['average'] = points if nn.configurations['average'] is None else nn.configurations['average'] * 0.99 + points * 0.01


	print("\n\nGame: ", nn.configurations['episodes'])
	print("Score this game:", points, "Average reward:", nn.configurations['average'])


	nn.compute_gradients(states, actions, rewards)
	
	if nn.configurations['episodes'] % nn.configurations['episodePerTrain'] == 0:
		print("Updating...")
		nn.total_update()
		nn.save()


	print()