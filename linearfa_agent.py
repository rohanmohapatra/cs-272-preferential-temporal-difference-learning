import random
import gym
import numpy as np


class LinearFAAgent:
    def __init__(self, env: gym.Env, n_features: int, gamma: float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5_000) -> None:
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
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.total_epi = total_epi
        self.n_features = n_features
        self.w = np.zeros((env.action_space.n, self.n_features))
        self.loss = []

    def eps_greedy(self, state: np.ndarray, exploration: bool = True) -> int:
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
                return np.argmax(np.dot(self.w, state))
        else:
            return np.argmax(np.dot(self.w, state))

    def greedy(self, ss: np.ndarray) -> int:
        return self.eps_greedy(ss, False)

    def choose_action(self, ss: np.ndarray) -> int:
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
        while not done and steps<=max_steps:
            s_state, reward, done, _, _ = self.env.step(action)
            episode.append((state, action, reward))
            action = self.greedy(s_state)

            state = s_state
            steps += 1
            # if steps == max_steps:
            #     return (episode, False)

        return (episode, True)

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

    def rmse(self):
        return np.sqrt(np.mean(np.array(self.loss)))


class SARSAAgent(LinearFAAgent):
    def learn(self) -> None:
        """SARSA algorithm
        Update the weights (self.w) for self.total_epi number of episodes.
        """
        # Loop for each episode:
        for each_episode in range(self.total_epi):
            if each_episode % 100 == 0:
                print(each_episode)
            # Initialize S
            state, _ = self.env.reset()
            action = self.choose_action(state)
            done = False

            while not done:
                state_dash, reward, done, _, _ = self.env.step(action)
                action_dash = self.choose_action(state_dash)
                # SARSA: Q(S, A) = Q(S, A) + alpha*(R + gamma*Q(S', A') - Q(S, A))
                delta = reward + self.gamma * \
                    np.dot(state_dash, self.w[action_dash]
                           ) - np.dot(state, self.w[action])

                self.loss.append(delta ** 2)

                self.w[action] = self.w[action] + \
                    self.alpha * delta * np.array(state)
                state = state_dash
                action = action_dash


class SARSALambdaAgent(LinearFAAgent):
    def __init__(self, env: gym.Env, n_features: int, lambd_a: float = 0.2, gamma: float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5000) -> None:
        super().__init__(env, n_features, gamma, eps, alpha, total_epi)
        self.lambd_a = lambd_a

    def learn(self) -> None:
        """SARSA Lambda algorithm
        Update the weights (self.w) for self.total_epi number of episodes.
        """
        # Loop for each episode:
        for each_episode in range(self.total_epi):
            if each_episode % 100 == 0:
                print(each_episode)
            # Initialize S
            state, _ = self.env.reset()
            action = self.choose_action(state)
            done = False
            e = 0
            while not done:
                state_dash, reward, done, _, _ = self.env.step(action)
                action_dash = self.choose_action(state_dash)

                # SARSA: Q(S, A) = Q(S, A) + alpha*(R + gamma*Q(S', A') - Q(S, A))
                delta = reward + self.gamma * \
                    np.dot(state_dash, self.w[action_dash]
                           ) - np.dot(state, self.w[action])

                e = self.gamma * self.lambd_a * e + np.array(state)

                self.loss.append((delta) ** 2)

                self.w[action] = self.w[action] + \
                    self.alpha * delta * e
                state = state_dash
                action = action_dash


class PTDAgent(LinearFAAgent):
    def __init__(self, env: gym.Env, n_features: int, beta: float = 0.2, gamma: float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5000) -> None:
        super().__init__(env, n_features, gamma, eps, alpha, total_epi)
        self.beta = beta

    def learn(self) -> None:
        """SARSA Lambda algorithm
        Update the weights (self.w) for self.total_epi number of episodes.
        """
        # Loop for each episode:
        for each_episode in range(self.total_epi):
            if each_episode % 100 == 0:
                print(each_episode)
            # Initialize S
            state, _ = self.env.reset()
            action = self.choose_action(state)
            done = False
            e = 0
            while not done:
                state_dash, reward, done, _, _ = self.env.step(action)
                action_dash = self.choose_action(state_dash)

                # SARSA: Q(S, A) = Q(S, A) + alpha*(R + gamma*Q(S', A') - Q(S, A))
                delta = reward + self.gamma * \
                    np.dot(state_dash, self.w[action_dash]
                           ) - np.dot(state, self.w[action])

                e = self.beta * np.array(state) + \
                    self.gamma * (1 - self.beta) * e

                self.loss.append((delta) ** 2)

                self.w[action] = self.w[action] + \
                    self.alpha * delta * e
                state = state_dash
                action = action_dash
