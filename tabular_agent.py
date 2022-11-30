from typing import Any
import random
import gym
import time


def argmax_action(d: dict[Any, float]) -> Any:
    """return a key of the maximum value in a given dictionary 

    Args:
        d (dict[Any,float]): dictionary

    Returns:
        Any: a key
    """
    return max(d, key=d.get)


class ValueRLAgent():
    def __init__(self, env: gym.Env, gamma: float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5_000) -> None:
        """initialize agent parameters
        This class will be a parent class and not be called directly.

        Args:
            env (gym.Env): gym environment
            gamma (float, optional): a discount factor. Defaults to 0.98.
            eps (float, optional): the epsilon value. Defaults to 0.2. Note: this pa uses a simple eps-greedy not decaying eps-greedy.
            alpha (float, optional): a learning rate. Defaults to 0.02.
            total_epi (int, optional): total number of episodes an agent should learn. Defaults to 5_000.
        """
        self.env = env
        self.q = self.init_qtable(env.observation_space.n, env.action_space.n)
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.total_epi = total_epi

    def init_qtable(self, n_states: int, n_actions: int, init_val: float = 0.0) -> dict[int, dict[int, float]]:
        """initialize the q table (dictionary indexed by s, a) with a given init_value

        Args:
            n_states (int, optional): the number of states. Defaults to int.
            n_actions (int, optional): the number of actions. Defaults to int.
            init_val (float, optional): all q(s,a) should be set to this value. Defaults to 0.0.

        Returns:
            dict[int,dict[int,float]]: q table (q[s][a] -> q-value)
        """
        q = dict()
        for state in range(n_states):
            q[state] = dict()
            for action in range(n_actions):
                q[state][action] = init_val

        return q

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """epsilon greedy algorithm to return an action

        Args:
            state (int): state
            exploration (bool, optional): explore based on the epsilon value if True; take the greedy action by the current self.q if False. Defaults to True.

        Returns:
            int: action
        """
        decider = random.random()
        if exploration:
            if decider < self.eps:
                return self.env.action_space.sample()
            else:
                return argmax_action(self.q[state])
        else:
            return argmax_action(self.q[state])

    def greedy(self, ss: int) -> int:
        return self.eps_greedy(ss, False)

    def choose_action(self, ss: int) -> int:
        """a helper function to specify a exploration policy [No change needed]

        Args:
            ss (int): state

        Returns:
            int: action
        """
        return self.eps_greedy(ss)

    def best_run(self, max_steps: int = 100) -> tuple[list[tuple[int, int, float]], bool]:
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
        action = self.greedy(state)
        done = False
        steps = 0
        episode = []
        while not done:
            s_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            action = self.greedy(s_state)

            state = s_state
            steps += 1
            if steps == max_steps:
                return (episode, False)

        return (episode, done)

    def calc_return(self, episode: list[tuple[int, int, float]], done=False) -> float:
        """Given an episode, calculate the return value. An episode is in this format: [(s1,a1,r1), (s2,a2,r2), ...].

        Args:
            episode (list[tuple[int,int,float]]): An episode [(s1,a1,r1), (s2,a2,r2), ...]
            done (bool, optional): True if the episode reaches a goal, False if it does not. Defaults to False.

        Returns:
            float: the return value. None if done is False.
        """
        if not done:
            return None

        return sum([pow(self.gamma, index) * reward for index, (_, _, reward) in enumerate(episode)])


class MCCAgent(ValueRLAgent):
    def learn(self) -> None:
        """Monte Carlo Control algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        """
        for each_episode in range(self.total_epi):
            state, _ = self.env.reset()
            action = self.choose_action(state)
            done = False
            episode = []
            max_steps = 50_000
            steps = 0
            while not done and steps < max_steps:
                state_dash, reward, done, _, _ = self.env.step(action)
                episode.append((state, action, reward))
                action = self.choose_action(state_dash)
                state = state_dash
                steps += 1

            g_t = 0
            for s, a, reward in reversed(episode):
                g_t = reward + self.gamma*g_t
                # MC: Q(s,a) = Q(s, a) + alpha*(G_t - Q(s,a))
                self.q[s][a] = self.q[s][a] + \
                    self.alpha * (g_t - self.q[s][a])


class SARSAAgent(ValueRLAgent):
    def learn(self) -> None:
        """SARSA algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        """
        # Loop for each episode:
        for each_episode in range(self.total_epi):
            # Initialize S
            state, _ = self.env.reset()
            action = self.choose_action(state)
            done = False

            while not done:
                state_dash, reward, done, _, _ = self.env.step(action)
                action_dash = self.choose_action(state_dash)

                # SARSA: Q(S, A) = Q(S, A) + alpha*(R + gamma*Q(S', A') - Q(S, A))
                self.q[state][action] += self.alpha * \
                    (reward + self.gamma * self.q[state_dash][action_dash]
                     - self.q[state][action])
                state = state_dash
                action = action_dash


class QLAgent(SARSAAgent):
    def learn(self):
        """Q-Learning algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        """
        # Loop for each episode:
        for each_episode in range(self.total_epi):
            # Initialize S
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.eps_greedy(state)
                state_dash, reward, done, _, _ = self.env.step(action)
                max_action = self.greedy(state_dash)

                # Q-Learning : Q(S, A) = Q(S, A) + alpha*(R + gamma* max_a Q(S', a) - Q(S, A))
                self.q[state][action] += self.alpha * \
                    (reward + self.gamma * self.q[state_dash][max_action]
                     - self.q[state][action])
                state = state_dash

    def choose_action(self, ss: int) -> int:
        """
        [optional] You may want to override this method.
        """
        pass


class SARSALambdaAgent(SARSAAgent):
    def __init__(self, env: gym.Env, lambd_a: float = 0.0, gamma: float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5000) -> None:
        super().__init__(env, gamma, eps, alpha, total_epi)
        self.lambd_a = lambd_a

        e = dict()
        for state in range(env.observation_space.n):
            e[state] = dict()
            for action in range(env.action_space.n):
                e[state][action] = 0
        self.e = e

    def learn(self) -> None:
        # Loop for each episode:
        for each_episode in range(self.total_epi):
            # Initialize S
            state, _ = self.env.reset()
            action = self.choose_action(state)
            done = False

            while not done:
                state_dash, reward, done, _, _ = self.env.step(action)
                action_dash = self.choose_action(state_dash)

                # SARSA: Q(S, A) = Q(S, A) + alpha*(R + gamma*Q(S', A') - Q(S, A))

                delta = reward + self.gamma * \
                    self.q[state_dash][action_dash] - self.q[state][action]
                self.e[state][action] = self.e[state][action] + 1
                for s, a_set in self.q.items():
                    for a in a_set:
                        self.q[s][a] = self.q[s][a] + \
                            self.alpha * delta * self.e[s][a]
                        self.e[s][a] = self.gamma * self.lambd_a * self.e[s][a]
                state = state_dash
                action = action_dash


class PTDAgent(SARSAAgent):
    def __init__(self, env: gym.Env, beta: float = 0.1, gamma: float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5000) -> None:
        super().__init__(env, gamma, eps, alpha, total_epi)
        self.beta = beta

        e = dict()
        for state in range(env.observation_space.n):
            e[state] = dict()
            for action in range(env.action_space.n):
                e[state][action] = 0
        self.e = e

    def learn(self) -> None:
        # Loop for each episode:
        for each_episode in range(self.total_epi):
            # Initialize S
            state, _ = self.env.reset()
            action = self.choose_action(state)
            done = False

            while not done:
                state_dash, reward, done, _, _ = self.env.step(action)
                action_dash = self.choose_action(state_dash)

                # SARSA: Q(S, A) = Q(S, A) + alpha*(R + gamma*Q(S', A') - Q(S, A))

                delta = reward + self.gamma * \
                    self.q[state_dash][action_dash] - self.q[state][action]
                self.e[state][action] = self.e[state][action] + self.beta
                for s, a_set in self.q.items():
                    for a in a_set:
                        self.q[s][a] = self.q[s][a] + \
                            self.alpha * delta * self.e[s][a]
                        self.e[s][a] = self.gamma * \
                            (1 - self.beta) * self.e[s][a]
                state = state_dash
                action = action_dash
