import gym
from lunar_landing_NeuralNetwork import *

game = 'LunarLander-v2'
env = gym.make(game)
wins = 0
nn = neuralNetwork(env.observation_space.shape[0], env.action_space.n)
for episode in range(1000):
	observation = env.reset()
	while True:
		if episode >= 100:
			env.render()
		nn.memoryState.append(observation)
		action = nn.prediction(observation)
		nn.memoryAction.append(action)
		observation, reward, done, info = env.step(action)
		print(observation)
		input()
		# if episode >= 100 and episode % 50 == 0:
		# 	print(observation)
		# 	print(reward)
		# 	print(done)
		# 	input('waiting...')
		nn.memoryNextState.append(observation)
		if done:
			if reward != 100 or reward != -100:
				nn.memoryReward.append(-200)
			else:
				nn.memoryReward.append(reward)
			break
		nn.memoryReward.append(reward)
	if wins == 0 and episode % 3 == 0:
		print(episode)
		nn.save('NN_model_'+game+'_5.h5')
		nn.replay(episode)
	print()
