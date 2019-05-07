import cv2, numpy, itertools, gym, time


def resize(img, size):
	dim = (size, size)
	img = img[30:]
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
	resized = numpy.divide(resized, 255)
	# cv2.imshow('image',img)
	# cv2.waitKey(0)
	return resized.tolist()



def discount_and_normalize_rewards(nn, episode_rewards):
	discounted_episode_rewards = numpy.zeros_like(episode_rewards)
	cumulative = 0.0
	for i in reversed(range(len(episode_rewards))):
		cumulative = cumulative * nn.configurations['discount'] + episode_rewards[i]
		discounted_episode_rewards[i] = cumulative
	
	mean = numpy.mean(discounted_episode_rewards)
	std = numpy.std(discounted_episode_rewards)
	discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

	return discounted_episode_rewards



def play_game(nn, render=False):
	
	env = gym.make(nn.configurations['game'])
	game_state_size = nn.configurations['observations']

	states, actions, rewards = [], [], []

	observation_list = [[[0]*game_state_size]*game_state_size]
	observation = resize(env.reset(), game_state_size)
	observation_list.append(observation)
	observation = list(itertools.chain.from_iterable(observation_list))
	states.append(observation)

	points = 0
	done = False

	while True:
		if render == True:
			env.render()
			# if (nn.configurations['episodes']) % nn.configurations['episodePerTrain'] == 0:
			#     # env.render()
			#     time.sleep(.01)

		action = nn.predict(observation)
		actions.append(action)
		
		observation, reward, done, info = env.step(action)
		rewards.append(reward)
		points += reward

		if done:
			break

		observation = resize(observation, game_state_size)
		observation_list.pop(0)
		observation_list.append(observation)
		observation = list(itertools.chain.from_iterable(observation_list))
		states.append(observation)


	env.close()
	nn.configurations['episodes'] += 1
	rewards = discount_and_normalize_rewards(nn, rewards)
	
	return states, actions, rewards, points