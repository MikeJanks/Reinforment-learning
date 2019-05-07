import os
import numpy as np
import gym
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from pg import Policy



def plot_graph_disc_and_gan(average_score_list):
	plt.figure(1)
	plt.clf()
	plt.ylim(-21, 21)
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
	image_dim = (64, 64)
	learning_rate = 0.00025
	hidden_size = 200
	action_size = 3
	gamma = 0.99
	batch_size = 10 # every how many episodes to do a param update?
	cuda_visible_devices = "0"
	gpu_memory_fraction = 0.3
	render = True
	load_model = False


	tf.reset_default_graph()

	policy = Policy(input_size=image_dim, hidden_size=hidden_size, action_size=action_size, gamma=gamma, learning_rate=learning_rate)

	env = gym.make("Pong-v0")

	prev_x = None         # used in computing the difference frame
	running_reward = None # moving average of episode rewards
	running_steps = None  # moving average of episode length
	steps = 0             # number of steps in the current episode
	total_steps = 0
	reward_sum = 0
	episode_number = 0
	num_of_games = 10000
	highest_avg_score = -99
	average_score_list = []


	os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())

		if load_model == True:
			policy.saver.restore(sess, './saves/')


		policy.init_gradient_buffer(sess)

		while episode_number<num_of_games:
		
			observation = env.reset()
			reward_sum = 0
			steps = 0
			ep_history = []
			done = False

			# create a difference image as starting image
			cur_x = preprocess(observation, image_dim)
			state = cur_x - np.zeros(image_dim)
			prev_x = cur_x

			while not done:

				if render: env.render()

				# sample an action from the policy
				action = policy.predict([state], sess) # action probability: the probability of moving UP

				# execute the action
				pong_action = {
					0: 0,
					1: 2,
					2: 3
				}
				# pong_action = 2 if action == 1 else 3 # 2=UP, 3=DOWN
				observation, reward, done, info = env.step(pong_action[action])

				last_state = state

				# create a difference image as network input
				cur_x = preprocess(observation, image_dim)
				state = cur_x - prev_x 
				prev_x = cur_x

				# append the step to the episode history
				ep_history.append([last_state, action, reward, state])
				reward_sum += reward
				steps += 1
				total_steps += 1

				if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
					print('reward: '+str(reward) + ('' if reward == -1 else ' !!!!!!!!'))


			# When episode is done
			episode_number += 1
			
			# book-keeping
			running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
			print('episode %d: episode_reward=%f mean_reward=%f' % (episode_number, reward_sum, running_reward), end=' ')

			if running_reward > highest_avg_score:
				highest_avg_score = running_reward
				try:
					os.mkdir('./saves/')
				except: pass
				policy.saver.save(session, './saves/')


			average_score_list.append(running_reward)
			ep_history = np.array(ep_history)

			# compute the discounted reward backwards through time
			discounted_epr = discount_rewards(ep_history[:, 2], gamma)


			# compute gradients
			p_loss, v_loss, grads = policy.compute_gradients(ep_history[:,0].tolist(), ep_history[:,1], ep_history[:, 2], discounted_epr, ep_history[:,0].tolist(), sess)
			print('p_loss=%f v_loss=%f episode_len=%i\n' % (p_loss, v_loss, steps))

			# collect the gradients until we update the network
			policy.collect_gradients(grads)

			# perform parameter update every batch_size episodes
			if episode_number % batch_size == 0 and episode_number != 0:
				print('Update policy\n')
				policy.update(sess)
				plot_graph_disc_and_gan(average_score_list)




if __name__ == "__main__":
	main()