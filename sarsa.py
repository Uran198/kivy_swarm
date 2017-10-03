import numpy as np
from collections import defaultdict

from env import Env, Action, ActionType
import pickle


class Sarsa:

    def __init__(self):
        self.feature_len = 42
        self.actions = [Action(ActionType.MOVE, di) for di in range(6)]

        self.params = np.array([0.1] * self.feature_len)
        self.step_size = 0.01

        self.reset_learning()

    def feature(self, state, action):
        res = [0] * self.feature_len
        # if state.food > 0:
        #     res[0] = 1 if state.neighbors[action.di].scent > 0 else 0
        # else:
        #     res[0] = state.neighbors[action.di].food
        n = len(state.neighbors) + 1
        k = action.di
        for i, sn in enumerate(state.neighbors):
            if state.food > 0:
                res[k * n + i] = sn.scent
            else:
                res[k * n + i] = sn.food
        # res[k * n + n - 1] = state.food
        return res

    def value_action_f(self, state, action):
        feature = self.feature(state, action)
        return np.dot(feature, self.params)

    def policy(self, state, explore=True):
        """Returns an action to take from state."""
        eps = self.exp_prob(state)
        if explore and np.random.choice([True, False], p=[eps, 1 - eps]):
            action = np.random.choice(self.actions)
        else:
            ind = np.argmax([self.value_action_f(state, a)
                            for a in self.actions])
            action = self.actions[ind]
        return action

    def alpha(self, feature, action):
        visited = self.vis_state_action[feature][action.di]
        if visited == 0:
            return 0
        return 1 / visited

    def exp_prob(self, state):
        """Returns the probability of random exploration."""
        return 0.05
        # return 150 / (150 + self.vis_state[feature])

    def learn(self, env, num_episodes, lmda):
        for time in range(num_episodes):
            state = env.reset()
            self.reset_learning(lmda)
            done = False
            tot_return = 0
            while not done:
                action = self.policy(state)
                next_state, reward, done = env.step(action)
                tot_return += reward

                self.adapt_policy(state, action, next_state, reward)

                state = next_state
            if time % 100 == 0:
                print(tot_return)
                print(self.params)
                print(self.feature(state, action))

    def reset_learning(self, lmda=0.5):
        '''
        Resets the learning scheme. In terms of td learning resets e trace.
        '''
        self.e_trace = np.array([0] * self.feature_len)
        self.lmda = lmda

    def adapt_policy(self, state, action, next_state, reward):
        ''' Makes correction to the policy. '''
        next_action = self.policy(next_state)
        feature = self.feature(state, action)

        self.e_trace = self.e_trace * self.lmda + feature
        ret = reward + self.value_action_f(next_state, next_action)
        td_error = ret - self.value_action_f(state, action)

        self.params += self.step_size * td_error * self.e_trace


def main():
    sarsa = Sarsa()
    env = Env(num_steps=200)
    sarsa.learn(env, 3000, 0.5)
    with open('sarsa.pickle', 'wb') as fd:
        pickle.dump(sarsa, fd)


if __name__ == "__main__":
    main()
