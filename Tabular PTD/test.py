import gym
import random

from main import *

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    td = TDlambda(env, 0.1)
    td.learn()
    epi, done = td.best_run()


    print(f'Best episode: {epi}, Return: {td.calc_return(epi, done)}')

    ptd = PreferentialTD(env, 0.1)
    ptd.learn()
    epi, done = ptd.best_run()


    print(f'Best episode: {epi}, Return: {ptd.calc_return(epi, done)}')