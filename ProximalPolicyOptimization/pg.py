import random, copy, os
from tqdm import tqdm
import numpy as np
import tensorflow as tf

class Policy():
	'''Policy Function approximator'''
	
	def __init__(self, input_size, action_size, clip=.2, _lambda=1, gamma=0.99, learning_rate=0.001, entropy_beta=0.01, val_discount=1):
		tf.reset_default_graph()
		os.environ["CUDA_VISIBLE_DEVICES"] = '0'
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.3

		self._lambda = _lambda
		self.gamma = gamma
		self.state_input = tf.placeholder(shape=[None, input_size[0], input_size[1]], dtype=tf.float32, name='state_input')
		# self.both_state_input = tf.placeholder(shape=[None, input_size[0], input_size[1]], dtype=tf.float32, name='state_input')
		# both_states = tf.stack([self.both_state_input], axis=-1)
		state = tf.stack([self.state_input], axis=-1)

		with tf.variable_scope("main_policy"):
			self.conv1 = tf.layers.conv2d(
				inputs=state,
				filters=32,
				kernel_size=3,
				strides=2,
				padding="same",
				activation=tf.nn.relu,
				name='conv1')

			self.conv2 = tf.layers.conv2d(
				inputs=self.conv1,
				filters=16,
				kernel_size=3,
				strides=2,
				padding="same",
				activation=tf.nn.relu,
				name='conv2')
				
			self.flat = tf.layers.flatten(self.conv2)
			# self.curr_flat, _ = tf.split(self.flat, 2)


			self.m_pg_fc1 = tf.layers.dense(
				inputs = self.flat,
				units = 200,
				activation=tf.nn.relu,
				name='m_pg_fc1')

			self.m_pg_fc2 = tf.layers.dense(
				inputs = self.m_pg_fc1,
				units = 200,
				activation=tf.nn.relu,
				name='m_pg_fc2')

			self.m_pg_logits = tf.layers.dense(
				inputs = self.m_pg_fc2,
				units = action_size,
				name='m_pg_logits')

			self.m_pg_probs = tf.nn.softmax(self.m_pg_logits)
			self.print_m_pg_logits = tf.print(self.m_pg_logits)

		self.max_action = tf.argmax(self.m_pg_probs)
		self.action = tf.multinomial(self.m_pg_logits, 1)


		with tf.variable_scope("old_policy"):
			self.o_conv1 = tf.layers.conv2d(
				inputs=state,
				filters=32,
				kernel_size=3,
				strides=2,
				padding="same",
				activation=tf.nn.relu,
				name='o_conv1')

			self.o_conv2 = tf.layers.conv2d(
				inputs=self.o_conv1,
				filters=16,
				kernel_size=3,
				strides=2,
				padding="same",
				activation=tf.nn.relu,
				name='o_conv2')
				
			self.o_flat = tf.layers.flatten(self.o_conv2)
			# self.curr_flat, _ = tf.split(self.flat, 2)

		
			self.o_pg_fc1 = tf.layers.dense(
				inputs = self.o_flat,
				units = 200,
				activation=tf.nn.relu,
				name='o_pg_fc1')

			self.o_pg_fc2 = tf.layers.dense(
				inputs = self.o_pg_fc1,
				units = 200,
				activation=tf.nn.relu,
				name='o_pg_fc2')

			self.o_pg_logits = tf.layers.dense(
				inputs = self.o_pg_fc2,
				units = action_size,
				name='o_pg_logits')

			self.o_pg_probs = tf.nn.softmax(self.o_pg_logits)


		with tf.variable_scope("main_value"):
			self.val_fc1 = tf.layers.dense(
				inputs = self.flat,
				units = 200,
				activation=tf.nn.relu,
				name='val_fc1')

			self.val_fc2 = tf.layers.dense(
				inputs = self.val_fc1,
				units = 200,
				activation=tf.nn.relu,
				name='val_fc2')

			self.val_logits = tf.layers.dense(
				inputs = self.val_fc2, 
				units = 1,
				name='val_logits')

			# curr_val, next_val = tf.split(self.val_logits, 2)
			self.curr_val = tf.reshape(self.val_logits, [-1])
			# self.next_val = tf.reshape(next_val, [-1])
		


		# Define the training procedure. 
		self.assign_ops = []
		for p_old, p in zip(tf.trainable_variables('old_policy'), tf.trainable_variables('main_policy')):
			self.assign_ops.append(tf.assign(p_old, p))
		self.action_holder 				= tf.placeholder(shape=[None], dtype=tf.int32, name='action_holder')
		self.reward_holder 				= tf.placeholder(shape=[None], dtype=tf.float32, name='reward_holder')
		self.next_value_holder 			= tf.placeholder(shape=[None], dtype=tf.float32, name='next_value_holder')
		self.gae_holder 				= tf.placeholder(shape=[None], dtype=tf.float32, name='gae_holder')

		# ------------------------------------------------------------------------------------------------#
		# Policy Loss																					  #
		# ------------------------------------------------------------------------------------------------#
		self.actions 		= tf.one_hot(self.action_holder, action_size)
		self.main_outputs 	= tf.reduce_sum(self.m_pg_probs * self.actions, axis=1)
		self.main_outputs 	= tf.clip_by_value(self.main_outputs,1e-10,1.0)
		self.old_outputs 	= tf.reduce_sum(self.o_pg_probs * self.actions, axis=1)
		self.old_outputs 	= tf.clip_by_value(self.old_outputs,1e-10,1.0)
		
		# self.trust 		= self.main_outputs/self.old_outputs
		self.trust 		= tf.exp(tf.log(self.main_outputs) - tf.stop_gradient(tf.log(self.old_outputs)), name='trust')
		self.cliped		= tf.clip_by_value(self.trust, 1-clip, 1+clip, name='clip')
		self.min		= tf.minimum(tf.multiply(self.trust, self.gae_holder), tf.multiply(self.cliped, self.gae_holder), name='min')
		self.pg_loss 	= -tf.reduce_mean(self.min, name='pg_loss')



		# ------------------------------------------------------------------------------------------------#
		# Value Loss																					  #
		# ------------------------------------------------------------------------------------------------#
		# self.val_loss = tf.square(self.curr_val - self.reward_holder)
		# self.val_loss = tf.losses.mean_squared_error(self.reward_holder, self.curr_val)
		self.val_loss = tf.square(self.curr_val - self.reward_holder + self.gamma * self.next_value_holder)
		# self.val_loss = tf.losses.mean_squared_error(self.reward_holder + self.gamma * self.next_value_holder, self.curr_val)
		self.val_loss = tf.reduce_mean(self.val_loss)
		# self.val_loss = val_discount * self.val_loss



		# ------------------------------------------------------------------------------------------------#
		# Entropy Loss																					  #
		# ------------------------------------------------------------------------------------------------#
		self.entropy 		= -tf.reduce_mean(tf.reduce_sum(self.m_pg_probs * tf.log(tf.clip_by_value(self.m_pg_probs,1e-10,1.0)), axis=1))
		self.entropy_loss 	= entropy_beta * self.entropy



		# ------------------------------------------------------------------------------------------------#
		# Total Loss																					  #
		# ------------------------------------------------------------------------------------------------#
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
		self.loss = tf.reduce_sum(self.pg_loss + self.val_loss - self.entropy_loss)

		self.train_batch = optimizer.minimize(self.loss)
		# gradients, variables = zip(*optimizer.compute_gradients(self.loss)
		# gradients = [ tf.where(tf.is_nan(x), tf.zeros_like(x), x) for x in gradients ]
		# # gradients, _ = tf.clip_by_global_norm(gradients, 5)
		# self.train_batch = optimizer.apply_gradients(zip(gradients, variables))

		self.saver = tf.train.Saver()
		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())
		



	def predict(self, x):
		feed_dict = { self.state_input: x }
		action, value = self.sess.run([self.action, self.curr_val], feed_dict=feed_dict)
		return action[0][0], value[0]


	def get_vals(self, x):
		feed_dict = { self.state_input: x }
		v_pred = self.sess.run(self.curr_val, feed_dict=feed_dict)
		return v_pred


	def play(self, x):
		feed_dict = { self.state_input: x }
		action = self.sess.run(self.max_action, feed_dict=feed_dict)[0]
		return action


	def get_gaes(self, rewards, v_preds, v_preds_next, normalize=False):
		gaes = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
		# calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
		# gaes = copy.deepcopy(deltas)
		for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
			gaes[t] = gaes[t] + self._lambda * self.gamma * gaes[t + 1]
		gaes = np.reshape(gaes, -1)
		# if normalize == True:
		# gaes = (gaes - gaes.mean()) / gaes.std()
		
		return gaes

		
	def train(self, transitions, epochs=4, batch_size=64, verbose=True):
		self.sess.run(self.assign_ops) 
		pg_loss_total_list, val_loss_total_list = [], []
		for _ in range(epochs):
			# print('-'*10, 'epoch'+str(_+1), '-'*10)
			pg_loss_list, val_loss_list = [], []
			random.shuffle(transitions)
			if verbose:
				tqdm_batchs = tqdm([transitions[i:i + batch_size] for i in range(0, len(transitions), batch_size)])
			else:
				tqdm_batchs	= [transitions[i:i + batch_size] for i in range(0, len(transitions), batch_size)]

			for batch in tqdm_batchs:
				states, actions, rewards, next_val, gaes = list(zip(*batch))
				feed_dict = {
					self.state_input: states,				# states
					self.action_holder: actions,  			# sampled action
					self.reward_holder: rewards,   			# reward
					self.next_value_holder: next_val,   	# value of next state
					self.gae_holder: gaes,   				# gaes
				}
				pg_loss, val_loss, _ = self.sess.run([self.pg_loss, self.val_loss, self.train_batch], feed_dict=feed_dict)
				pg_loss_list.append(pg_loss)
				val_loss_list.append(val_loss)
				if verbose:
					tqdm_batchs.desc = 'P:%f  V:%f' % (np.average(pg_loss_list), np.average(val_loss_list))
			pg_loss_total_list.append(np.average(pg_loss_list))
			val_loss_total_list.append(np.average(val_loss_list))
		if not verbose:
			print('P:%f  V:%f' % (np.average(pg_loss_total_list), np.average(val_loss_total_list)))
