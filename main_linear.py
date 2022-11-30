import gym
import random
from linearfa_agent import *

if __name__ == '__main__':
    random.seed(200)

    environment = gym.make('CartPole-v1')
    sarsa_agent = SARSAAgent(environment, 4)
    sarsa_agent.learn()

    episode, done = sarsa_agent.best_run()
    print(
        f'SARSA Best episode: {len(episode)}, return: {sarsa_agent.calc_return(episode, done)}, rmse: {sarsa_agent.rmse()}')

    environment = gym.make('CartPole-v1')
    slambda_agent = SARSALambdaAgent(environment, 4)
    slambda_agent.learn()

    episode, done = slambda_agent.best_run()
    print(
        f'SARSA Lambda Best episode: {len(episode)}, return: {slambda_agent.calc_return(episode, done)}, rmse: {slambda_agent.rmse()}')

    environment = gym.make('CartPole-v1')
    ptd_agent = PTDAgent(environment, 4)
    ptd_agent.learn()

    episode, done = ptd_agent.best_run()
    print(
        f'PTD Best episode: {len(episode)}, return: {ptd_agent.calc_return(episode, done)}, rmse: {ptd_agent.rmse()}')
