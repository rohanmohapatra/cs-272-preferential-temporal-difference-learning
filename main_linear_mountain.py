import gym
import random
from linearfa_agent import *

if __name__ == '__main__':
    random.seed(100)

    environment = gym.make('MountainCar-v0')
    sarsa_agent = SARSAAgent(environment, 2)
    sarsa_agent.learn()

    episode, done = sarsa_agent.best_run(1000)
    print(
        f'SARSA Best episode: {len(episode)}, return: {sarsa_agent.calc_return(episode, done)}, rmse: {sarsa_agent.rmse()}')

    environment = gym.make('MountainCar-v0')
    slambda_agent = SARSALambdaAgent(environment, 2)
    slambda_agent.learn()

    episode, done = slambda_agent.best_run(1000)
    print(
        f'SARSA Lambda Best episode: {len(episode)}, return: {slambda_agent.calc_return(episode, done)}, rmse: {slambda_agent.rmse()}')

    environment = gym.make('MountainCar-v0')
    ptd_agent = PTDAgent(environment, 2)
    ptd_agent.learn()

    episode, done = ptd_agent.best_run(1000)
    print(
        f'PTD Best episode: {len(episode)}, return: {ptd_agent.calc_return(episode, done)}, rmse: {ptd_agent.rmse()}')
