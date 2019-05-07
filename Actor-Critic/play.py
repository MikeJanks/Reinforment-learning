import os
import numpy as np
import gym
import tensorflow as tf
import cv2
from pg import Policy


def preprocess(img, image_dim):
	'''Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector'''
	img = img[30:]
	resized = cv2.resize(img, image_dim, interpolation = cv2.INTER_AREA)
	resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
	resized = np.divide(resized, 255)
	# cv2.imshow('image',img)
	# cv2.waitKey(0)
	return resized


def main():

	image_dim = (64, 64)
	hidden_size = 200
	action_size = 2
	cuda_visible_devices = "1"
	gpu_memory_fraction = 0.3
	render = True
	load_model = False


	tf.reset_default_graph()

	policy = Policy(input_size=image_dim, hidden_size=hidden_size, action_size=action_size)

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

		policy.saver.restore(sess, './saves/')
		

		while True:

			if render: env.render()

			# create a difference image as network input
			cur_x = preprocess(observation, image_dim)
			x = cur_x - prev_x if prev_x is not None else np.zeros(image_dim)
			prev_x = cur_x

			# sample an action from the policy
			action = policy.predict([x], sess)

			# execute the action
			pong_action = 2 if action == 0 else 3
			observation, reward, done, info = env.step(pong_action)
			
			reward_sum += reward
			steps += 1
			total_steps += 1

		
			if done: # an episode finished

				episode_number += 1
				
				# book-keeping
				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				running_steps = steps if running_steps is None else running_steps * 0.99 + steps * 0.01
				print('episode %d: episode_reward=%f mean_reward=%f\n' % (episode_number, reward_sum, running_reward))

				observation = env.reset()
				reward_sum = 0
				prev_x = None
				steps = 0


if __name__ == "__main__":
	main()