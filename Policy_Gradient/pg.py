import tensorflow as tf

class PolicyEstimator():
	'''Policy Function approximator'''
	
	def __init__(self, input_size, hidden_size, action_size, learning_rate=0.001):

		self.state_input = tf.placeholder(shape=[None, input_size[0], input_size[1]], dtype=tf.float32, name='state_input')
		state = tf.stack([self.state_input], axis=-1)

		with tf.variable_scope("policy"):
			self.conv1 = tf.layers.conv2d(inputs=state,
				filters=32,
				kernel_size=3,
				strides=2,
				padding="same", 
				activation=tf.nn.relu, 
				name='conv1')

			self.conv2 = tf.layers.conv2d(inputs=self.conv1,
				filters=16,
				kernel_size=3,
				strides=2,
				padding="same", 
				activation=tf.nn.relu,
				name='conv2')

			self.flat = tf.layers.flatten(self.conv2)
			self.fc1 = tf.layers.dense(
				inputs = self.flat, 
				units = hidden_size, 
				activation=tf.nn.relu,
				name='fc1')

			self.fc2 = tf.layers.dense(
				inputs = self.fc1, 
				units = hidden_size, 
				activation=tf.nn.relu,
				name='fc2')

			self.logits = tf.layers.dense(
				inputs = self.fc2, 
				units = action_size,
				name='logits')

			self.probs = tf.nn.softmax(self.logits)

		self.max_action = tf.argmax(self.probs)
		self.action = tf.multinomial(self.logits, 1)
		
		# Define the training procedure. 
		
		self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
		self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)


		self.actions = tf.one_hot(self.action_holder, action_size)
		self.responsible_outputs = tf.reduce_sum(self.probs * self.actions, axis=1)

		self.advantage = self.reward_holder
		self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.advantage)


		tvars = tf.trainable_variables("policy")
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


	def play(self, x, sess):
		feed_dict = { self.state_input: x }
		action = sess.run(self.max_action, feed_dict=feed_dict)[0]
		return action


	def compute_gradients(self, states, actions, rewards, sess):
		feed_dict = {
			self.state_input: states,     # begin state
			self.action_holder: actions,  # sampled action
			self.reward_holder: rewards   # discounted reward
		}
		p_loss, grads = sess.run([self.loss, self.gradients], feed_dict=feed_dict)
		return p_loss, grads


	def init_gradient_buffer(self, sess):
		self.grad_buffer = sess.run(tf.trainable_variables("policy"))
		self._reset_gradient_buffer()
	

	def collect_gradients(self, gradients):
		for idx, grad in enumerate(gradients):
			self.grad_buffer[idx] += grad
			
		
	def update(self, session):
		feed_dict = dict(zip(self.gradient_holders, self.grad_buffer))
		session.run(self.update_batch, feed_dict=feed_dict)
		self._reset_gradient_buffer()
		try:
			os.mkdir('./saves/')
		except: pass
		self.saver.save(session, './saves/')


	def _reset_gradient_buffer(self):
		for ix, grad in enumerate(self.grad_buffer):
			self.grad_buffer[ix] = grad * 0

