import gym
import random

from main import *

if __name__ == '__main__':
    random.seed(1010)
    env = gym.make('CliffWalking-v0')
    x = [0.2, 0.4, 0.6, 0.8, 1]

    y_td = []
    y_ptd = []

    for i in range(len(x)):
        td = TDlambda(env, x[i])
        td.learn()
        epi, done = td.best_run()
        y_td.append(td.calc_return(epi, done))

        ptd = PreferentialTD(env, x[i])
        ptd.learn()
        epi2, done2 = ptd.best_run()
        y_ptd.append(ptd.calc_return(epi2, done2))

    print(y_td)
    print(y_ptd)
    