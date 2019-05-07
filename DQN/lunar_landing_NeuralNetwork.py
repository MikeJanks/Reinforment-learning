import numpy
import random
# from pynput.keyboard import Key, Controller
import tensorflow as tf
from pprint import pprint

class neuralNetwork:
	def __init__(self, observations, actions):
		# self.keyboard = Controller()
		self.memoryState    	= []
		self.memoryNextState    = []
		self.memoryAction   	= []
		self.memoryReward   	= []
		# self.currReward     	= []
		self.batch_size     	= 64
		self.discount       	= .99
		self.epsilon        	= .9
		self.epsilonDecay   	= .05
		self.episodePerTrain	= 1
		self.actions			= actions
		self.observations		= observations

		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=2e-4), input_shape=(observations,)))
		self.model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=2e-4)))
		# self.model.add(tf.keras.layers.Dense(units=actions, activation=tf.nn.softmax, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=2e-3)))
		self.model.add(tf.keras.layers.Dense(units=actions, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=2e-3)))
		self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
		# self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.01, decay=.99))

	def prediction(self, state):
		if random.random() <= self.epsilon:
			return random.randrange(0, self.actions)
		else:
			predict = self.model.predict(numpy.array([state]))
			return numpy.argmax(predict[0])

	def save(self, filepath):
		self.model.save(filepath)


	def replay(self, episode):
		# # Set reward values for current state
		# listlen = len(self.currReward)-1
		# final_reward = self.currReward[-1]
		# for i in range(len(self.currReward)):
		# 	reward = (self.discount**(listlen-i))*final_reward
		# 	self.memoryReward.append(reward)
		# self.currReward = []

		# # Set raw reward values from current rewards
		# for reward in self.currReward:
		# 	self.memoryReward.append(reward)
		# self.currReward = []

		# Get main Memory list
		memory_size = len(self.memoryState)
		print('Memory List:', memory_size)

		# check if memory has enough in list
		if memory_size < self.batch_size or ((episode+1)%self.episodePerTrain) != 0:
			return

		for _ in range(1):
			states = []
			target_vecs = []
			print('Shuffling...')
			rand_list = list(range(len(self.memoryState)))
			random.shuffle(rand_list)
			print('Getting Targets...')
			for i in rand_list:
				next_state_predict = self.model.predict(numpy.array([self.memoryNextState[i]]))[0]
				target = self.memoryReward[i]+self.discount*numpy.amax(next_state_predict)
				target_vec = self.model.predict(numpy.array([self.memoryState[i]]))[0]
				target_vec[self.memoryAction[i]] = target
				states.append(self.memoryState[i])
				target_vecs.append(target_vec)
			print('Training...')
			self.model.fit(numpy.array(states), numpy.array(target_vecs), batch_size=self.batch_size, verbose=1)

		if self.epsilon > .1:
			self.epsilon -= self.epsilonDecay
		
