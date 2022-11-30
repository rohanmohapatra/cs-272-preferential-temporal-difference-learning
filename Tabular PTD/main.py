from typing import Any
import random
import gym


def argmax_action(d: dict[Any,float]) -> Any:
    """return a key of the maximum value in a given dictionary 

    Args:
        d (dict[Any,float]): dictionary

    Returns:
        Any: a key
    """
    max_value = float('-inf')
    max_i = -1
    for i in d:
        if d[i] > max_value:
            max_value = d[i]
            max_i = i
    
    return max_i

class ValueRLAgent():
    def __init__(self, env: gym.Env, gamma : float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5_000) -> None:
        """initialize agent parameters
        This class will be a parent class and not be called directly.

        Args:
            env (gym.Env): gym environment
            gamma (float, optional): a discount factor. Defaults to 0.98.
            eps (float, optional): the epsilon value. Defaults to 0.2. Note: this pa uses a simple eps-greedy not decaying eps-greedy.
            alpha (float, optional): a learning rate. Defaults to 0.02.
            total_epi (int, optional): total number of episodes an agent should learn. Defaults to 5_000.
        """
        print(env)
        self.env = env
        self.q = self.init_qtable(env.observation_space.n, env.action_space.n)
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.total_epi = total_epi
    
    def init_qtable(self, n_states: int, n_actions: int, init_val: float = 0.0) -> dict[int,dict[int,float]]:
        """initialize the q table (dictionary indexed by s, a) with a given init_value

        Args:
            n_states (int, optional): the number of states. Defaults to int.
            n_actions (int, optional): the number of actions. Defaults to int.
            init_val (float, optional): all q(s,a) should be set to this value. Defaults to 0.0.

        Returns:
            dict[int,dict[int,float]]: q table (q[s][a] -> q-value)
        """
        q = dict()
        for i in range(n_states):
            row = dict()
            for j in range(n_actions):
                row[j] = 0

            q[i] = row
        
        return q

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """epsilon greedy algorithm to return an action

        Args:
            state (int): state
            exploration (bool, optional): explore based on the epsilon value if True; take the greedy action by the current self.q if False. Defaults to True.

        Returns:
            int: action
        """
        max_a = argmax_action(self.q[state])
        if exploration == False:
            return max_a

        p = random.random()

        if p > self.eps:
            return max_a
        else:
            return self.env.action_space.sample()

        

        pass
    
    def choose_action(self, ss: int) -> int:
        """a helper function to specify a exploration policy [No change needed]

        Args:
            ss (int): state

        Returns:
            int: action
        """
        return self.eps_greedy(ss)         
    
    def best_run(self, max_steps: int = 100) -> tuple[list[tuple[int,int,float]], bool]:
        """After the learning, an optimal episode (based on the latest self.q) needs to be generated for evaluation. From the initial state, always take the greedily best action until it reaches a goal.

        Args:
            max_steps (int, optional): Terminate the episode generation if the agent cannot reach the goal after max_steps. One step is (s,a,r) Defaults to 100.

        Returns:
            tuple[
                list[tuple[int,int,float]],: An episode [(s1,a1,r1), (s2,a2,r2), ...]
                bool: done - True if the episode reaches a goal, False if it hits max_steps.
            ]
        """
        state, _ = self.env.reset()
        steps = 0
        episode = []
        done = False
        
        while steps != max_steps and done != True:
            action = self.eps_greedy(state, exploration=False)
            next_step = self.env.step(action)
            steps = steps + 1
            reward = next_step[1]
            episode.append((state, action, reward))
            state = next_step[0]
            done = next_step[2]
        
        return (episode, done)


    def calc_return(self, episode: list[tuple[int,int,float]], done=False) -> float:
        """Given an episode, calculate the return value. An episode is in this format: [(s1,a1,r1), (s2,a2,r2), ...].

        Args:
            episode (list[tuple[int,int,float]]): An episode [(s1,a1,r1), (s2,a2,r2), ...]
            done (bool, optional): True if the episode reaches a goal, False if it does not. Defaults to False.

        Returns:
            float: the return value. None if done is False.
        """
        if done == False:
            return None
        
        ep_return  = 0
        iter = 1

        for step in episode:
            ep_return = ep_return + pow(self.gamma, iter - 1)*step[2]
            iter = iter + 1
        
        return ep_return


        pass       

class MCCAgent(ValueRLAgent):
    def learn(self) -> None:
        """Monte Carlo Control algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        """
        iter = 0

        while iter < self.total_epi: 
            epsiode = []
            s, _ = self.env.reset()
            
            done = False
            while not done:
                a = self.choose_action(s)
                ss, r, done, _, _ = self.env.step(a)
                epsiode.append((s, a, r))
                s = ss
                
            Gt = 0
            for exp in reversed(epsiode):
                s = exp[0]
                a = exp[1]
                Gt = float(exp[2]) + self.gamma*Gt
                self.q[s][a] = self.q[s][a] + self.alpha*(Gt - self.q[s][a])

            iter = iter + 1
            
        return

class TDlambda(ValueRLAgent):
    def __init__(self, env: gym.Env, Lambda:float,  gamma : float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5_000):
        self.Lambda = Lambda
        ValueRLAgent.__init__(self, env, gamma, eps, alpha, total_epi)
   
    def init_e_table(self, n_states: int, n_actions: int) -> dict[int,dict[int,float]]:
        """initialize the q table (dictionary indexed by s, a) with a given init_value

        Args:
            n_states (int, optional): the number of states. Defaults to int.
            n_actions (int, optional): the number of actions. Defaults to int.
            init_val (float, optional): all q(s,a) should be set to this value. Defaults to 0.0.

        Returns:
            dict[int,dict[int,float]]: q table (q[s][a] -> q-value)
        """
        E = dict()
        for i in range(n_states):
            row = dict()
            for j in range(n_actions):
                row[j] = 0

            E[i] = row
        
        return E

    def learn(self) -> None:

        iter = 0

        while iter < self.total_epi:
            if iter%1000 == 0:
                print(iter)
            E = self.init_e_table(self.env.observation_space.n, self.env.action_space.n)
            S, _ = self.env.reset()
            A = self.choose_action(S)
            done = False

            while not done:
                S_dash, R, done, _, _ = self.env.step(A)
                A_dash = self.choose_action(S_dash)
                error = R + self.gamma*self.q[S_dash][A_dash] - self.q[S][A]
                E[S][A] = E[S][A] + 1
                for s in range(self.env.observation_space.n):
                    for a in range(self.env.action_space.n):
                        self.q[s][a] = self.q[s][a] + self.alpha*error*E[s][a]
                        E[s][a] = self.gamma*self.Lambda*E[s][a]
                
                S = S_dash
                A = A_dash
            
            iter = iter + 1


class PreferentialTD(ValueRLAgent):
    def __init__(self, env: gym.Env, beta: float, gamma : float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5_000):
        self.beta = beta
        ValueRLAgent.__init__(self, env, gamma, eps, alpha, total_epi)

    def init_e_table(self, n_states: int, n_actions: int) -> dict[int,dict[int,float]]:
        """initialize the q table (dictionary indexed by s, a) with a given init_value

        Args:
            n_states (int, optional): the number of states. Defaults to int.
            n_actions (int, optional): the number of actions. Defaults to int.
            init_val (float, optional): all q(s,a) should be set to this value. Defaults to 0.0.

        Returns:
            dict[int,dict[int,float]]: q table (q[s][a] -> q-value)
        """
        E = dict()
        for i in range(n_states):
            row = dict()
            for j in range(n_actions):
                row[j] = 0

            E[i] = row
        
        return E

    def learn(self) -> None:

        iter = 0

        while iter < self.total_epi:
            if iter%1000 == 0:
                print(iter)
            E = self.init_e_table(self.env.observation_space.n, self.env.action_space.n)
            S, _ = self.env.reset()
            A = self.choose_action(S)
            done = False

            while not done:
                S_dash, R, done, _, _ = self.env.step(A)
                A_dash = self.choose_action(S_dash)
                error = R + self.gamma*self.q[S_dash][A_dash] - self.q[S][A]
                E[S][A] = E[S][A] + self.beta
                for s in range(self.env.observation_space.n):
                    for a in range(self.env.action_space.n):
                        self.q[s][a] = self.q[s][a] + self.alpha*error*E[s][a]
                        E[s][a] = self.gamma*(1 - self.beta)*E[s][a]
                
                S = S_dash
                A = A_dash
            
            iter = iter + 1
