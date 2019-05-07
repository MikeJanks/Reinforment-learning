import random, os, pickle
from pprint import pprint
from tqdm import tqdm
import numpy, random
import tensorflow as tf


class neuralNetwork:

	# ============================================================== #
	# Helper Functions                                               #
	# ============================================================== #

	def predict(self, observation):
		game_state_size = self.configurations['observations']
		feed_dict={
			self.observations: [numpy.array(observation).reshape([game_state_size*2, game_state_size, 1])]
		}
		return self.sess.run(self.predict_action, feed_dict=feed_dict)[0][0]

	def compute_gradients(self, states, actions, rewards):
		game_state_size 		= self.configurations['observations']
		game 					= list(zip(states, actions, rewards))
		random.shuffle(game)
		batchs 					= [game[i:i + self.configurations['policy_batch_size']] for i in range(0, len(game), self.configurations['policy_batch_size'])]
		for batch in tqdm(batchs):
			states, actions, rewards = list(zip(*batch))
			feed_dict = {
				self.observations: numpy.array(states).reshape([len(states), game_state_size*2, game_state_size, 1]),
				self.actions: actions,
				self.rewards: rewards
			}
			value_grads, pg_grads_list 	= self.sess.run([self.value_gradients, self.pg_gradients], feed_dict=feed_dict)
			self.value_grad_buffer 	= [self.value_grad_buffer[idx] + grad for idx, grad in enumerate(value_grads)]
			for pg_grads in list(zip(*pg_grads_list)):
				self.pg_grad_buffer 	= [self.pg_grad_buffer[idx] + grad for idx, grad in enumerate(pg_grads)]

	def init_session_and_network(self, new):
		self.sess = tf.Session()
		init = tf.global_variables_initializer()
		self.sess.run(init)
		if not new:
			self.saver.restore(self.sess, './saves/'+self.configurations['game']+'_'+self.configurations['model_number']+'/')
		self.pg_grad_buffer 	= self.sess.run(self.pg_trainable_vars)
		self.value_grad_buffer 	= self.sess.run(self.value_trainable_vars)
		self._reset_gradient_buffer()

	def _reset_gradient_buffer(self):
		self.pg_grad_buffer 	= [ grad * 0 for grad in self.pg_grad_buffer ]
		self.value_grad_buffer 	= [ grad * 0 for grad in self.value_grad_buffer ]

	def total_update(self):
		feed_value_dict 			= dict(zip(self.value_gradient_holders, self.value_grad_buffer))
		feed_pg_dict 				= dict(zip(self.pg_gradient_holders, self.pg_grad_buffer))
		# self.sess.run(self.value_update, 	feed_dict=feed_value_dict)
		self.sess.run(self.pg_update, 		feed_dict=feed_pg_dict)
		self._reset_gradient_buffer()

	def save(self):
		try:
			os.mkdir('./saves/')
		except: pass
		try:
			os.mkdir('./saves/'+self.configurations['game']+'_'+self.configurations['model_number']+'/')
		except: pass
		self.saver.save(self.sess, './saves/'+self.configurations['game']+'_'+self.configurations['model_number']+'/')
		pickle.dump(self.configurations, open('./saves/'+self.configurations['game']+'_'+self.configurations['model_number']+'/config.pk', 'wb'))

	
	def __init__(self, game, observations, actions, model_number, new):
		print('\nGetting memory...')
		if new:
			self.configurations	= {}
			self.configurations['game']					= game
			self.configurations['episodes']				= 0
			self.configurations['average']				= None
			self.configurations['policy_batch_size']	= 64
			self.configurations['discount']				= 0.99
			self.configurations['episodePerTrain']		= 2
			self.configurations['actions']				= actions
			self.configurations['observations']			= observations
			self.configurations['model_number']			= model_number
			self.configurations['pg_epochs']			= 1
			self.configurations['policy_learning_rate']	= 0.00025
			self.configurations['value_learning_rate']	= 0.0005
		else:
			self.configurations = pickle.load(open('./saves/'+str(game)+'_'+str(model_number)+'/config.pk', 'rb'))

		print()
		pprint(self.configurations)
		print()
		
		print('Getting model...\n')
		
		self.observations = tf.placeholder(tf.float32, shape=[None, observations*2, observations, 1])
		self.actions = tf.placeholder(tf.int32, shape=[None])
		self.rewards = tf.placeholder(tf.float32, shape=[None])
		self.total = tf.placeholder(tf.float32, shape=None)


		# ============================================================== #
		# Main Network	                                                 #
		# ============================================================== #
		with tf.variable_scope("policy_network"):
			# features = tf.contrib.layers.layer_norm(self.observations)
			
			# conv1 = tf.layers.conv2d(inputs=features, filters=16, kernel_size=3, strides=2, padding="same")
			conv1 = tf.layers.conv2d(inputs=self.observations, filters=16, kernel_size=3, strides=2, padding="same")
			conv1_act = tf.nn.leaky_relu(conv1)

			conv2 = tf.layers.conv2d(inputs=conv1_act, filters=32, kernel_size=3, strides=2, padding="same")
			conv2_act = tf.nn.leaky_relu(conv2)

			# flat = tf.layers.flatten(features)
			# flat = tf.layers.flatten(self.observations)
			flat = tf.layers.flatten(conv2_act)
			dense1 = tf.layers.dense(inputs=flat, units=200)
			dense1_act = tf.nn.leaky_relu(dense1)


		with tf.variable_scope("policy"):
			dense2 = tf.layers.dense(inputs=dense1_act, units=200)
			dense2_act = tf.nn.leaky_relu(dense2)

			logits = tf.layers.dense(inputs=dense2_act, units=self.configurations['actions'])


		with tf.variable_scope("value"):
			value = tf.layers.dense(inputs=dense1_act, units=1)
			value = tf.reshape(value, [-1])


		self.predict_action = tf.multinomial(logits, 1)
		

		# ============================================================== #
		# Training                                                       #
		# ============================================================== #
		value_optimizer = tf.train.AdamOptimizer(learning_rate=self.configurations['value_learning_rate'])

		self.value_trainable_vars = tf.trainable_variables("value")
		value_loss = tf.losses.mean_squared_error(self.rewards, value)
		# value_loss = tf.reduce_sum(value_loss)/self.total
		# value_loss = tf.reduce_sum(value_loss)/self.total
		self.value_gradients = tf.gradients(value_loss, self.value_trainable_vars)

		self.value_gradient_holders = []
		for idx, var in enumerate(self.value_trainable_vars):
			placeholder = tf.placeholder(tf.float32, name=str(idx) + '_value_holder')
			self.value_gradient_holders.append(placeholder)
		self.value_update = value_optimizer.apply_gradients(zip(self.value_gradient_holders, self.value_trainable_vars))


		policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.configurations['policy_learning_rate'])
		# policy_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0005, decay=0.99, momentum=0.0, epsilon=1e-6)

		self.pg_trainable_vars = tf.trainable_variables("policy")
		neg_log_prob = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.actions)
		pg_loss = neg_log_prob * (self.rewards)
		# pg_loss = tf.reduce_sum(neg_log_prob * (self.rewards-value))
		# pg_loss = tf.reduce_sum(neg_log_prob * (self.rewards-value))/self.total
		# pg_loss = tf.reduce_mean(neg_log_prob * (self.rewards))
		# self.pg_gradients = policy_optimizer.compute_gradients(pg_loss)
		self.pg_gradients = tf.gradients(pg_loss, self.pg_trainable_vars)

		self.pg_gradient_holders = []
		for idx, var in enumerate(self.pg_trainable_vars):
			placeholder = tf.placeholder(tf.float32, name=str(idx) + '_pg_holder')
			self.pg_gradient_holders.append(placeholder)
		self.pg_update = policy_optimizer.apply_gradients(zip(self.pg_gradient_holders, self.pg_trainable_vars))
		
		self.saver = tf.train.Saver()

		# ============================================================== #
		# Initialize and load                                            #
		# ============================================================== #
		self.init_session_and_network(new)

