import gym
import numpy
import sys
from tensorflow.keras.models import load_model


game = 'CartPole-v0'
env = gym.make(game)
nn = load_model('NN_model_CartPole-v0_'+str(sys.argv[1])+'.h5')
for episode in range(1000):
    observation = env.reset()
    for t in range(250):
        env.render()
        action = numpy.argmax(nn.predict(numpy.array([observation]))[0])
        observation, reward, done, info = env.step(action)
        if done:
            if t == 199:
                print('Win!')
            else:
                print('Lose!')
            break
    print(episode)
