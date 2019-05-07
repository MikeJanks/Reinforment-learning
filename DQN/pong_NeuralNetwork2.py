import numpy
import random
import pandas
import pickle
import tensorflow as tf
from pprint import pprint
from tqdm import tqdm
import ast

class neuralNetwork:
	def __init__(self, game, observations, actions, model_number, new):
		if new:
			print('\nGetting new memory...')
			self.model_memory = []
			self.configurations	= {}
			self.configurations['game']				= game
			self.configurations['episodes']			= 1
			self.configurations['maxMemorySize']	= 500000
			self.configurations['batch_size']		= 64
			self.configurations['discount']			= 0.99
			self.configurations['epsilonDecay']		= 0.99
			self.configurations['epsilon']			= self.configurations['epsilonDecay']**self.configurations['episodes']
			self.configurations['episodePerTrain']	= 1
			self.configurations['actions']			= actions
			self.configurations['observations']		= observations
			self.configurations['model_number']		= model_number
			self.configurations['epochs']			= 1
			self.configurations['learning_rate']	= 0.00001
			pickle.dump(self.configurations, open('.\\NeuralNetwork_saved_data\\'+game+'\\'+game+'_'+str(model_number)+'_config.pk', 'wb'))

			print('Getting new model...\n')
			self.model = tf.keras.Sequential()
			self.model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None), input_shape=(observations,)))
			self.model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None)))
			self.model.add(tf.keras.layers.Dense(units=actions, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None)))
			self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.configurations['learning_rate']))

		else:
			print('\nGetting old memory...')
			self.model_memory = pandas.read_csv('.\\NeuralNetwork_saved_data\\'+game+'\\'+game+'_'+str(model_number)+'_memory.csv').drop(columns=['Unnamed: 0']).values.tolist()
			for x in range(len(self.model_memory)):
				self.model_memory[x][0] = ast.literal_eval(self.model_memory[x][0])
				self.model_memory[x][3] = ast.literal_eval(self.model_memory[x][3])
			self.configurations = pickle.load(open('.\\NeuralNetwork_saved_data\\'+game+'\\'+game+'_'+str(model_number)+'_config.pk', 'rb'))

			print('Getting old model...\n')
			self.model = tf.keras.models.load_model('.\\NeuralNetwork_saved_data\\'+game+'\\'+game+'_'+str(model_number)+'_model.nn')


		self.stops = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=5)
		print()
		pprint(self.configurations)
		print()
		

	def prediction(self, state):
		if random.random() <= self.configurations['epsilon']:
			# if self.configurations['episodes'] % self.configurations['episodePerTrain'] == 0 and self.configurations['episodes'] > 50:
			# 	print('|          |Random!   |')
			return random.randrange(0, self.configurations['actions'])
		else:
			predict = self.model.predict(numpy.array([state]))
			# if self.configurations['episodes'] % self.configurations['episodePerTrain'] == 0 and self.configurations['episodes'] > 50:
			# 	print('|'+str(numpy.argmax(predict[0]))+'         |          |')
			return numpy.argmax(predict[0])


	def save(self, filepath):
		self.model.save(filepath)


	def replay(self):
		# Get main Memory list
		memory_size = len(self.model_memory)
		if memory_size > self.configurations['maxMemorySize']:
			for x in range(memory_size - self.configurations['maxMemorySize']):
				self.model_memory.pop(0)


		print('Memory List:', memory_size)
		self.configurations['episodes'] += 1

		# check if memory has enough in list
		if memory_size < self.configurations['batch_size'] or ((self.configurations['episodes']+1)%self.configurations['episodePerTrain']) != 0 or memory_size < 5000:
			return

		states = []
		target_vecs = []
		print('Shuffling...')
		random.shuffle(self.model_memory)
		print('Getting Targets...')

		for i in tqdm(range(len(self.model_memory))):
			state, action, reward, nextState = self.model_memory[i]
			target_vec = self.model.predict(numpy.array([state]))[0]
			next_state_predict = self.model.predict(numpy.array([nextState]))[0]
			target = reward + self.configurations['discount'] * numpy.amax(next_state_predict)
			target_vec[action] = target
			states.append(state)
			target_vecs.append(target_vec)

		print('Training...')
		self.model.fit(numpy.array(states), numpy.array(target_vecs), batch_size=self.configurations['batch_size'], verbose=1, epochs=self.configurations['epochs'])

		print('\nSaving...\nDO NOT STOP!\n')

		tf.keras.models.save_model(self.model,'.\\NeuralNetwork_saved_data\\'+self.configurations['game']+'\\'+self.configurations['game']+'_'+str(self.configurations['model_number'])+'_model.nn')
		pandas.DataFrame(self.model_memory).to_csv('.\\NeuralNetwork_saved_data\\'+self.configurations['game']+'\\'+self.configurations['game']+'_'+str(self.configurations['model_number'])+'_memory.csv')
		pickle.dump(self.configurations, open('.\\NeuralNetwork_saved_data\\'+self.configurations['game']+'\\'+self.configurations['game']+'_'+str(self.configurations['model_number'])+'_config.pk', 'wb'))


		if self.configurations['epsilon'] > .02:
			self.configurations['epsilon']=(self.configurations['epsilonDecay']**self.configurations['episodes'])
