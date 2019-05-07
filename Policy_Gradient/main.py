import os
import numpy as np
import gym
import tensorflow as tf
import cv2
from pg import PolicyEstimator




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
	action_size = 2
	gamma = 0.99
	batch_size = 10 # every how many episodes to do a param update?
	cuda_visible_devices = "1"
	gpu_memory_fraction = 0.3
	render = True
	load_model = False


	tf.reset_default_graph()

	policy = PolicyEstimator(input_size=image_dim, hidden_size=hidden_size, action_size=action_size, learning_rate=learning_rate)

	env = gym.make("Pong-v0")
	observation = env.reset()

	prev_x = None         # used in computing the difference frame
	running_reward = None # moving average of episode rewards
	running_steps = None  # moving average of episode length
	steps = 0             # number of steps in the current episode
	total_steps = 0
	reward_sum = 0
	episode_number = 0

	os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction

	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())

		if load_model == True:
			policy.saver.restore(sess, './saves/')


		policy.init_gradient_buffer(sess)

		ep_history = []

		while True:

			if render: env.render()

			# create a difference image as network input
			cur_x = preprocess(observation, image_dim)
			x = cur_x - prev_x if prev_x is not None else np.zeros(image_dim)
			prev_x = cur_x

			# sample an action from the policy
			action = policy.play([x], sess) # action probability: the probability of moving UP

			# execute the action
			pong_action = 2 if action == 0 else 3 # 2=UP, 3=DOWN
			observation, reward, done, info = env.step(pong_action)

			# append the step to the episode history
			ep_history.append([x, action, reward])
			reward_sum += reward
			steps += 1
			total_steps += 1

		
			if done: # an episode finished

				episode_number += 1

				ep_history = np.array(ep_history)

				# compute the discounted reward backwards through time
				discounted_epr = discount_rewards(ep_history[:, 2], gamma)


				# compute gradients
				p_loss, grads = policy.compute_gradients(ep_history[:,0].tolist(), ep_history[:,1], discounted_epr, sess)

				# collect the gradients until we update the network
				policy.collect_gradients(grads)

				# perform parameter update every batch_size episodes
				if episode_number % batch_size == 0 and episode_number != 0:
					print('Update policy')
					policy.update(sess)
				
				# book-keeping
				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				running_steps = steps if running_steps is None else running_steps * 0.99 + steps * 0.01
				print('episode %d: episode_reward=%f mean_reward=%f loss=%f episode_len=%i\n' % (episode_number-1, reward_sum, running_reward, p_loss, steps))

				observation = env.reset()
				reward_sum = 0
				prev_x = None
				steps = 0
				ep_history = []



			if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
				print('reward: '+str(reward) + ('' if reward == -1 else ' !!!!!!!!'))


if __name__ == "__main__":
	main()