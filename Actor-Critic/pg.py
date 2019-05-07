import numpy as np
import tensorflow as tf

class Policy():
	'''Policy Function approximator'''
	
	def __init__(self, input_size, hidden_size, action_size, gamma, learning_rate=0.001, entropy_beta=0.01):

		
		self.state_input = tf.placeholder(shape=[None, input_size[0], input_size[1]], dtype=tf.float32, name='state_input')
		self.both_state_input = tf.placeholder(shape=[None, input_size[0], input_size[1]], dtype=tf.float32, name='state_input')
		both_states = tf.stack([self.both_state_input], axis=-1)
		state = tf.stack([self.state_input], axis=-1)
		# both = tf.concat([state, next_state], 0)


		with tf.variable_scope("policy"):
			self.pg_conv1 = tf.layers.conv2d(
				inputs=state,
				filters=32,
				kernel_size=3,
				strides=2,
				padding="same", 
				activation=tf.nn.relu, 
				name='pg_conv1')

			self.pg_conv2 = tf.layers.conv2d(
				inputs=self.pg_conv1,
				filters=16,
				kernel_size=3,
				strides=2,
				padding="same", 
				activation=tf.nn.relu,
				name='pg_conv2')
				
			self.flat = tf.layers.flatten(self.pg_conv2)

			self.pg_fc1 = tf.layers.dense(
				inputs = self.flat,
				units = hidden_size,
				activation=tf.nn.relu,
				name='pg_fc1')

			self.pg_fc2 = tf.layers.dense(
				inputs = self.pg_fc1,
				units = hidden_size,
				activation=tf.nn.relu,
				name='pg_fc2')

			self.pg_logits = tf.layers.dense(
				inputs = self.pg_fc2,
				units = action_size,
				name='pg_logits')

			self.pg_probs = tf.nn.softmax(self.pg_logits)

		self.action = tf.multinomial(self.pg_logits, 1)


		with tf.variable_scope("value"):
			self.val_conv1 = tf.layers.conv2d(
				inputs=both_states,
				filters=32,
				kernel_size=3,
				strides=2,
				padding="same", 
				activation=tf.nn.relu, 
				name='val_conv1')

			self.val_conv2 = tf.layers.conv2d(
				inputs=self.val_conv1,
				filters=16,
				kernel_size=3,
				strides=2,
				padding="same", 
				activation=tf.nn.relu,
				name='val_conv2')

			self.both_flat = tf.layers.flatten(self.val_conv2)

			self.val_fc1 = tf.layers.dense(
				inputs = self.both_flat,
				units = hidden_size,
				activation=tf.nn.relu,
				name='val_fc1')

			self.val_fc2 = tf.layers.dense(
				inputs = self.val_fc1,
				units = hidden_size,
				activation=tf.nn.relu,
				name='val_fc2')

			self.val_logits = tf.layers.dense(
				inputs = self.val_fc2, 
				units = 1,
				name='val_logits')

			curr_val, next_val = tf.split(self.val_logits, 2)
			self.curr_val = tf.reshape(curr_val, [-1])
			self.next_val = tf.reshape(next_val, [-1])
		
		# Define the training procedure. 
		
		self.action_holder 				= tf.placeholder(shape=[None], dtype=tf.int32)
		action_indices 					= tf.stack([tf.range(tf.size(self.action_holder)), self.action_holder], axis=-1)
		self.reward_holder 				= tf.placeholder(shape=[None], dtype=tf.float32)
		self.discounted_rewards_holder 	= tf.placeholder(shape=[None], dtype=tf.float32)

		# ------------------------------------------------------------------------------------------------#
		# Policy Loss																					  #
		# ------------------------------------------------------------------------------------------------#
		self.actions = tf.one_hot(self.action_holder, action_size)
		self.responsible_outputs = tf.reduce_sum(self.pg_probs * self.actions, axis=1)

		# value_estimate 	= tf.gather_nd(self.val_logits, action_indices)
		# self.advantage 	= self.reward_holder-value_estimate
		self.advantage 	= self.discounted_rewards_holder + gamma*self.next_val - self.curr_val
		self.pg_loss 	= -tf.reduce_mean(tf.log(self.responsible_outputs) * self.advantage)
		self.pg_loss 	= 1e-8 if self.pg_loss == 0 else self.pg_loss

		# ------------------------------------------------------------------------------------------------#
		# Value Loss																					  #
		# ------------------------------------------------------------------------------------------------#
		# target_value 		= tf.Variable([], validate_shape=False)
		# target_value 		= tf.assign(target_value, self.val_logits, validate_shape=False)
		# target_value 		= tf.scatter_nd_update(target_value, action_indices, self.reward_holder)
		# self.val_loss 		= tf.losses.mean_squared_error(target_value, self.val_logits)
		self.val_loss 		= tf.losses.mean_squared_error(self.reward_holder, self.curr_val)
		# self.val_loss 		= 1e-8 if self.val_loss == 0 else self.val_loss



		# ------------------------------------------------------------------------------------------------#
		# Entropy Loss																					  #
		# ------------------------------------------------------------------------------------------------#
		self.entropy 		= -tf.reduce_mean(tf.reduce_sum(self.pg_probs * tf.log(self.pg_probs), axis=1))
		self.entropy_loss 	= entropy_beta * self.entropy
		# self.entropy_loss 	= 1e-8 if self.entropy_loss == 0 else self.entropy_loss



		# ------------------------------------------------------------------------------------------------#
		# Total Loss																					  #
		# ------------------------------------------------------------------------------------------------#
		self.loss = self.pg_loss + self.val_loss - self.entropy_loss
		# self.loss = 1e-8 if self.loss == 0 else self.loss
	
		tvars = tf.trainable_variables()
		self.gradients = tf.gradients(self.loss, tvars)
		
		self.gradient_holders = []
		for idx, var in enumerate(tvars):
			placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
			self.gradient_holders.append(placeholder)

		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

		self.saver = tf.train.Saver()



	def predict(self, x, sess):
		feed_dict = { self.state_input: x }
		action = sess.run(self.action, feed_dict=feed_dict)[0][0]
		return action


	def compute_gradients(self, states, actions, rewards, discounted_rewards, next_state, sess):
		feed_dict = {
			self.state_input: states,									# begin state
			self.both_state_input: np.concatenate([states,next_state]),	# both begin & next state
			self.action_holder: actions,  								# sampled action
			self.reward_holder: rewards,   								# reward
			self.discounted_rewards_holder: discounted_rewards,   		# discounted reward
		}
		p_loss, v_loss, grads = sess.run([self.pg_loss, self.val_loss, self.gradients], feed_dict=feed_dict)
		return p_loss, v_loss, grads


	def init_gradient_buffer(self, sess):
		self.grad_buffer = sess.run(tf.trainable_variables())
		self._reset_gradient_buffer()
	

	def collect_gradients(self, gradients):
		for idx, grad in enumerate(gradients):
			self.grad_buffer[idx] += grad
			
		
	def update(self, session):
		feed_dict = dict(zip(self.gradient_holders, self.grad_buffer))
		session.run(self.update_batch, feed_dict=feed_dict)
		self._reset_gradient_buffer()


	def _reset_gradient_buffer(self):
		for ix, grad in enumerate(self.grad_buffer):
			self.grad_buffer[ix] = grad * 0

