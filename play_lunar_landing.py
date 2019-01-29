import gym
import numpy
import sys
from tensorflow.keras.models import load_model


game = 'LunarLander-v2'
env = gym.make(game)
nn = load_model('.\\NeuralNetwork_saved_data\\NN_model_LunarLander-v2_'+str(sys.argv[1])+'.h5')
wins = 0
for episode in range(1000):
    observation = env.reset()
    while(True):
        env.render()
        action = numpy.argmax(nn.predict(numpy.array([observation]))[0])
        observation, reward, done, info = env.step(action)
        if done:
            if reward == 100:
                wins += 1
                print('Win!')
            else:
                print('Lose!')
            print(reward)
            break
    print(episode)
    temp = episode + 1
    print(wins/(temp))
    print()