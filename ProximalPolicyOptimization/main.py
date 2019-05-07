import sys, os, pickle, gym, cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pg import Policy
from multiprocessing import Process, Manager




def plot_graph(average_score_list):
	plt.figure(1)
	plt.clf()
	plt.ylim(-21, 21)
	plt.xlim(0, 10000)
	plt.plot(average_score_list, c='b')
	plt.savefig('./Graph.png')


def discount_rewards(r, gamma):
	'''Take float array of rewards and compute discounted reward'''
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(0, r.size)):
		# if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
		
	discounted_r -= np.mean(discounted_r)
	discounted_r /= np.std(discounted_r)

	return discounted_r


def preprocess(img, image_dim):
	img = img[30:]
	resized = cv2.resize(img, image_dim, interpolation = cv2.INTER_AREA)
	resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
	resized = np.divide(resized, 255)
	# cv2.imshow('image',img)
	# cv2.waitKey(0)
	return resized



def main():
	print('Continue?(Y,n)', end=' ')
	settings = {
		'game': "Pong-v0",
		'image_dim': (64, 64),
		'average_score_list': [],
		'curr_episode': 1,
		'running_reward': None
	}
	load_model = True if input().lower() == 'y' else False
	learning_rate = 0.00025
	action_size = 3
	update_num = 512
	game_step = 0
	states, actions, rewards, vals, gaes, temp_states, temp_rewards, temp_vals = [], [], [], [], [], [], [], []
	transitions = []


	
	policy = Policy(input_size=settings['image_dim'], action_size=action_size, learning_rate=learning_rate)

	if load_model == True:
		policy.saver.restore(policy.sess, './saves/')
		settings = pickle.load(open('./saves/info.continue', 'rb'))
		settings['curr_episode']+=1
		
	
	env = gym.make(settings['game'])

	# while episode_number<num_of_games:
	for episode_number in range(settings['curr_episode'], 10000):
		
		# policy.sess.run(policy.assign_ops)
		observation = env.reset()
		reward_sum = 0
		done = False

		# create a difference image as starting image
		cur_x = preprocess(observation, settings['image_dim'])
		state = cur_x - np.zeros(cur_x.shape)
		prev_x = cur_x

		while not done:
			if True: env.render()

			game_step+=1

			# sample an action from the policy
			action, value = policy.predict([state])

			# execute the action
			pong_action = { 0: 0, 1: 2, 2: 3 }
			observation, reward, done, info = env.step(pong_action[action])

			last_state = state

			# create a difference image as network input
			cur_x = preprocess(observation, settings['image_dim'])
			state = cur_x - prev_x 
			prev_x = cur_x


			# append the step to the episode lists
			states.append(last_state)
			actions.append(action)
			rewards.append(reward)
			vals.append(value)
			temp_states.append(last_state)
			temp_rewards.append(reward)
			temp_vals.append(value)
			

			reward_sum += reward
			
			
			# perform parameter update every update_num episodes
			if (game_step % update_num == 0 and game_step != 0):
				settings['curr_episode'] = episode_number
				next_vals = vals[1:] + policy.get_vals([state])
				temp_next_vals = temp_vals[1:] + policy.get_vals([state])
				gaes = gaes + policy.get_gaes(temp_rewards, temp_vals, temp_next_vals).tolist()
				gaes = np.array(gaes)
				gaes = (gaes - gaes.mean()) / gaes.std()
				policy.train(list(zip(states, actions, rewards, next_vals, gaes)), epochs=3, batch_size=128, verbose=False)
				states, actions, rewards, vals, gaes, temp_states, temp_rewards, temp_vals, = [], [], [], [], [], [], [], []




		# When episode is done
		# book-keeping
		settings['running_reward'] = reward_sum if settings['running_reward'] is None else settings['running_reward'] * 0.99 + reward_sum * 0.01
		print('\nPPO4\nepisode %d: episode_reward=%f mean_reward=%f \n' % (episode_number, reward_sum, settings['running_reward']))
		settings['average_score_list'].append(settings['running_reward'])

		temp_next_vals = temp_vals[1:] + policy.get_vals([state])
		gaes = gaes + policy.get_gaes(temp_rewards, temp_vals, temp_next_vals).tolist()
		temp_states, temp_rewards, temp_vals, = [], [], []
		# gaes = policy.get_gaes(rewards, vals, next_vals)
		
		# print('\nUpdate policy')
		settings['curr_episode'] = episode_number
		# policy.train(list(zip(states, actions, rewards, next_vals, gaes)), epochs=4, batch_size=64)
		try:
			os.mkdir('./saves/')
		except: pass
		policy.saver.save(policy.sess, './saves/')
		pickle.dump(settings, open('./saves/info.continue', 'wb'))
		plot_graph(settings['average_score_list'])
		transitions = []
		# states, actions, rewards, vals = [], [], [], []
	
		print()


if __name__ == "__main__":
	main()