from copy import deepcopy
import numpy as np
from collections import defaultdict

from env import Env, Action, ActionType


class Sarsa:

    def __init__(self):
        # FIXME
        # Assume for now that food in each hex can be at most 1.
        # Represent state as 6 integers represing food in different dirs. Where
        # agent stand food is always 0.
        self.features = list(range(1 << 6))
        self.actions = [Action(ActionType.MOVE, di) for di in range(6)]
        self.value_action = [[0 for _ in self.actions] for _ in self.features]
        self.vis_state_action = deepcopy(self.value_action)
        self.vis_state = [0 for _ in self.features]

    def feature(self, state):
        res = 0
        for d, sn in enumerate(state.neighbors):
            if sn.food > 0:
                res += (1 << d)
        return res

    def policy(self, state):
        """Returns an action to take from state."""
        f = self.feature(state)
        eps = self.exp_prob(f)
        if np.random.choice([True, False], p=[eps, 1 - eps]):
            # explore
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
            action = self.policy(state)
            feature = self.feature(state)
            e_trace = defaultdict(lambda: 0)
            done = False
            while not done:
                self.visit(feature, action)
                next_state, reward, done = env.step(action)
                next_action = self.policy(next_state)
                next_feature = self.feature(next_state)

                ret = reward + self.value_action[next_feature][next_action.di]
                td_error = ret - self.value_action[feature][action.di]
                e_trace[feature, action] += 1
                for f in self.features:
                    for a in self.actions:
                        upd = self.alpha(f, a) * td_error * e_trace[f, a]
                        self.value_action[f][a.di] += upd
                        e_trace[f, a] *= lmda
                state = next_state
                action = next_action
                feature = next_feature


def main():
    sarsa = Sarsa()
    env = Env(num_steps=50)
    sarsa.learn(env, 400, 0.5)
    for act in sarsa.value_action:
        print(act)


if __name__ == "__main__":
    main()
