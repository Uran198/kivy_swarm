import numpy as np
from collections import defaultdict

from env import Env, Action, ActionType
import pickle


class Sarsa:

    def __init__(self):
        # FIXME
        # Assume for now that food in each hex can be at most 1.
        # Represent state as 6 integers represing food in different dirs. Where
        # agent stand food is always 0.
        self.features = list(range(1 << 7))
        self.actions = [Action(ActionType.MOVE, di) for di in range(6)]
        self.value_action = [[0 for _ in self.actions] for _ in self.features]
        self.vis_state_action = [[0 for _ in self.actions]
                                 for _ in self.features]
        self.vis_state = [0 for _ in self.features]
        self.reset_learning()

    def feature(self, state):
        res = 0
        if state.food == 0:
            for d, sn in enumerate(state.neighbors):
                if sn.food > 0:
                    res += (1 << d)
        else:
            for d, sn in enumerate(state.neighbors):
                if sn.scent >= 1:
                    res += (1 << d)
        res += state.food * (1 << 6)
        return res

    def policy(self, state, explore=True):
        """Returns an action to take from state."""
        f = self.feature(state)
        eps = self.exp_prob(f)
        if explore and np.random.choice([True, False], p=[eps, 1 - eps]):
            action = np.random.choice(self.actions)
        else:
            ind = np.argmax([self.value_action[f][a.di]
                            for a in self.actions])
            action = self.actions[ind]
        return action

    def alpha(self, feature, action):
        visited = self.vis_state_action[feature][action.di]
        if visited == 0:
            return 0
        return 1 / visited

    def exp_prob(self, feature):
        """Returns the probability of random exploration."""
        return 150 / (150 + self.vis_state[feature])

    def visit(self, feature, action):
        self.vis_state[feature] += 1
        self.vis_state_action[feature][action.di] += 1

    def learn(self, env, num_episodes, lmda):
        for time in range(num_episodes):
            state = env.reset()
            self.reset_learning(lmda)
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done = env.step(action)

                self.adapt_policy(state, action, next_state, reward)

                state = next_state

    @staticmethod
    def e_trace_factory():
        return 0

    def reset_learning(self, lmda=0.5):
        '''
        Resets the learning scheme. In terms of td learning resets e trace.
        '''
        self.e_trace = defaultdict(self.e_trace_factory)
        self.lmda = lmda

    def adapt_policy(self, state, action, next_state, reward):
        ''' Makes correction to the policy. '''
        next_action = self.policy(next_state)
        feature = self.feature(state)
        next_feature = self.feature(next_state)

        self.visit(feature, action)

        ret = reward + self.value_action[next_feature][next_action.di]
        td_error = ret - self.value_action[feature][action.di]
        self.e_trace[feature, action] += 1

        for f in self.features:
            for a in self.actions:
                upd = self.alpha(f, a) * td_error * self.e_trace[f, a]
                self.value_action[f][a.di] += upd
                self.e_trace[f, a] *= self.lmda


def main():
    sarsa = Sarsa()
    env = Env(num_steps=50)
    sarsa.learn(env, 400, 0.5)
    for fea, act in enumerate(sarsa.value_action):
        print("{:#08b}".format(fea), act)
    with open('sarsa.pickle', 'wb') as fd:
        pickle.dump(sarsa, fd)


if __name__ == "__main__":
    main()
