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
			self.model_memory = {}
			self.model_memory['memoryState']		= []
			self.model_memory['memoryNextState']	= []
			self.model_memory['memoryAction']		= []
			self.model_memory['memoryReward']		= []
			self.model_memory['episodes']			= 1
			self.configurations['game']				= game
			self.configurations['maxMemorySize']	= 500000
			self.configurations['batch_size']		= 128
			self.configurations['discount']			= 0.9
			self.configurations['epsilonDecay']		= 0.95
			self.configurations['epsilon']			= self.configurations['epsilonDecay']**self.model_memory['episodes']
			self.configurations['episodePerTrain']	= 1
			self.configurations['actions']			= actions
			self.configurations['observations']		= observations
			self.configurations['model_number']		= model_number
			self.configurations['epochs']			= 4
			self.configurations['learning_rate']	= 0.00025
			
		else:
			print('\nGetting old memory...')
			self.model_memory = pandas.read_csv('.\\NeuralNetwork_saved_data\\'+game+'_'+str(model_number)+'_memory.csv').drop(columns=['Unnamed: 0']).to_dict(orient='list')
			for x in range(len(self.model_memory['memoryState'])):
				self.model_memory['memoryState'][x] = ast.literal_eval(self.model_memory['memoryState'][x])
				self.model_memory['memoryNextState'][x] = ast.literal_eval(self.model_memory['memoryNextState'][x])
			self.model_memory['episodes'] = self.model_memory['episodes'][0]
			self.configurations = pickle.load(open('.\\NeuralNetwork_saved_data\\'+game+'_'+str(model_number)+'_config.pk', 'rb'))

		self.stops = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=5)

		if new:
			print('Getting new model...\n')
			self.model = tf.keras.Sequential()
			self.model.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None), input_shape=(observations,)))
			self.model.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None)))
			self.model.add(tf.keras.layers.Dense(units=actions, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None)))
			self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.configurations['learning_rate']))
		else:
			print('Getting old model...\n')
			self.model = tf.keras.models.load_model('.\\NeuralNetwork_saved_data\\'+game+'_'+str(model_number)+'_model.nn')



	def prediction(self, state):
		if random.random() <= self.configurations['epsilon']:
			if self.model_memory['episodes'] % self.configurations['episodePerTrain'] == 0 and self.model_memory['episodes'] > 50:
				print('|          |Random!   |')
			return random.randrange(0, self.actions)
		else:
			predict = self.model.predict(numpy.array([state]))
			if self.model_memory['episodes'] % self.configurations['episodePerTrain'] == 0 and self.model_memory['episodes'] > 50:
				print('|'+str(numpy.argmax(predict[0]))+'         |          |')
			return numpy.argmax(predict[0])

	def save(self, filepath):
		self.model.save(filepath)


	def replay(self):
		# Get main Memory list
		memory_size = len(self.model_memory['memoryAction'])
		if memory_size > self.configurations['maxMemorySize']:
			for x in range(memory_size - self.configurations['maxMemorySize']):
				self.model_memory['memoryState'].pop(0)
				self.model_memory['memoryNextState'].pop(0)
				self.model_memory['memoryAction'].pop(0)
				self.model_memory['memoryReward'].pop(0)
				


		print('Memory List:', memory_size)

		self.model_memory['episodes'] += 1

		# check if memory has enough in list
		if memory_size < self.batch_size or ((self.model_memory['episodes']+1)%self.episodePerTrain) != 0 or memory_size < 5000:
			return

		states = []
		target_vecs = []
		print('Shuffling...')
		rand_list = list(range(len(self.model_memory['memoryState'])))
		random.shuffle(rand_list)
		print('Getting Targets...')

		for i in tqdm(rand_list):
			next_state_predict = self.model.predict(numpy.array([self.model_memory['memoryNextState'][i]]))[0]
			target = self.model_memory['memoryReward'][i]+self.discount*numpy.amax(next_state_predict)
			target_vec = self.model.predict(numpy.array([self.model_memory['memoryState'][i]]))[0]
			target_vec[self.model_memory['memoryAction'][i]] = target
			states.append(self.model_memory['memoryState'][i])
			target_vecs.append(target_vec)
		print('Training...')
		epochs = int(memory_size/self.epochs)
		self.model.fit(numpy.array(states), numpy.array(target_vecs), batch_size=self.batch_size, verbose=1, epochs=epochs, callbacks=[self.stops])

		print('\nSaving...\nDO NOT STOP!\n')

		tf.keras.models.save_model(self.model,'.\\NeuralNetwork_saved_data\\pong_'+str(self.model_number)+'_model.nn')
		pandas.DataFrame(self.model_memory).to_csv('.\\NeuralNetwork_saved_data\\pong_'+str(self.model_number)+'_memory.csv')


		if self.epsilon > .05:
			self.epsilon=(self.epsilonDecay**self.model_memory['episodes'])
		
