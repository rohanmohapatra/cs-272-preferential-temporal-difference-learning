import gym
import random
from tabular_agent import *

if __name__ == '__main__':
    random.seed(1010)

    environment = gym.make('CliffWalking-v0')
    sarsa_agent = SARSAAgent(environment)
    sarsa_agent.learn()

    # Greedy route (based on the latest q-table)
    episode, done = sarsa_agent.best_run()
    print(
        f'Best episode: {episode}, return: {sarsa_agent.calc_return(episode, done)}')

    environment = gym.make('CliffWalking-v0')
    sarsa_lambda_agent = SARSALambdaAgent(environment, 0.4)
    sarsa_lambda_agent.learn()

    episode, done = sarsa_lambda_agent.best_run()
    print(
        f'Best episode: {episode}, return: {sarsa_lambda_agent.calc_return(episode, done)}')

    environment = gym.make('CliffWalking-v0')
    ptd_agent = PTDAgent(environment, 0.4)
    ptd_agent.learn()

    episode, done = ptd_agent.best_run()
    print(
        f'Best episode: {episode}, return: {ptd_agent.calc_return(episode, done)}')

    environment = gym.make('CliffWalking-v0')
    ql_agent = QLAgent(environment)
    ql_agent.learn()

    # Greedy route (based on the latest q-table)
    episode, done = ql_agent.best_run()
    print(
        f'Best episode: {episode}, return: {ql_agent.calc_return(episode, done)}')

    # mcc_agent = MCCAgent(environment)
    # mcc_agent.learn()

    # episode, done = mcc_agent.best_run()
    # print(
    #     f'Best episode: {episode}, return: {mcc_agent.calc_return(episode, done)}')
