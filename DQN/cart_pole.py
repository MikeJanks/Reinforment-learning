import gym
from cart_pole_NeuralNetwork import *

game = 'CartPole-v0'
wins = 0
finished = False
env = gym.make(game)
print(env.action_space.n)
print(env.observation_space.shape[0])
nn = neuralNetwork(env.observation_space.shape[0], env.action_space.n)
for episode in range(1000):
    observation = env.reset()
    for t in range(1000):
        if episode >= 150:
            env.render()
        nn.memoryState.append(observation)
        action = nn.prediction(observation)
        nn.memoryAction.append(action)
        observation, reward, done, info = env.step(action)
        nn.memoryNextState.append(observation)
        if done:
            print(t)
            if t == 199:
                print('Win!')
                wins += 1
                if wins >= 5:
                    nn.save('NN_model_'+game+'_3.h5')
                    finished = True
                nn.memoryReward.append(-1)
            else:
                print('Lose!')
                wins = 0
                nn.memoryReward.append(-1)
            break
        nn.memoryReward.append(0)
    print('Episode: '+str(episode))
    print('Wins: '+str(wins))
    if finished:
        break
    if wins == 0 and episode % 3 == 0:
        nn.replay(episode)
    # input('waiting...')
