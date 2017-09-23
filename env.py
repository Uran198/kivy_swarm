# TODO: some small unit tests maybe?
import numpy as np
from collections import namedtuple
from enum import Enum


# TODO: Is better way to call it observation?
State = namedtuple('State', ['cell', 'neighbors'])
State.__doc__ += '''
State contains cell in which agent is situated and all adjacent cells.
'''


Hex = namedtuple('Hex', ['q', 'r', 'food'])
Hex.__doc__ += '''
Contains information about the hex cell, like present food or trails.
'''


class ActionType(Enum):
    ''' All possible action types the agent can take. '''
    MOVE = object()


class Action(object):
    ''' All the actions the agent can take. '''

    def __init__(self, act_type, di=None):
        if not (0 <= di < 6):
            raise ValueError
        self.act_type = act_type
        if act_type is ActionType.MOVE:
            self.di = di

    def __repr__(self):
        return "Action(act_type={}, di={})".format(self.act_type, self.di)


class BaseGrid():
    '''
    Contains all neccessary information about the hex cells and maps them
    to the Axial coordinates. Inherit this class if you want to do some
    additional processing, like vizualization.
    '''
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def __init__(self, *args, **kwargs):
        self.cells = {}

    def init(self, q, r):
        ''' Initiates cell with coordinates q, r. '''
        food = np.random.choice([1, 0], p=[0.1, 0.9])
        self.cells[q, r] = Hex(q, r, food)

    def neighbor_dir(self, q, r, d):
        # This raise condition is not expected to be handled.
        assert 0 <= d < 6

        q += BaseGrid.dirs[d][0]
        r += BaseGrid.dirs[d][1]
        if (q, r) not in self.cells:
            self.init(q, r)
        return self.cells[q, r]

    def neighbors(self, q, r):
        for d in BaseGrid.dirs:
            yield self.neighbor_dir(q, r, d)

    def __setitem__(self, key, val):
        if len(key) != 2:
            raise KeyError
        self.cells[key] = val

    def __getitem__(self, key):
        if len(key) != 2:
            raise KeyError
        if key not in self.cells:
            self.init(*key)
        return self.cells[key]


class Env(object):
    ''' Environment with which the agent can interact. '''
    def __init__(self, grid_cls=BaseGrid):
        self.grid = grid_cls()
        self.grid_cls = grid_cls

    def step(self, action):
        '''
        Performs the action and returns new state, reward and whether the
        state is terminal.
        '''
        reward = 0
        if action.act_type is ActionType.MOVE:
            cell = self.grid.neighbor_dir(
                self.agent_pos[0], self.agent_pos[1], action.di)
            self.agent_pos = cell.q, cell.r
            neighbors = self.grid.neighbors(*self.agent_pos)
            reward = cell.food
            self.grid[cell.q, cell.r] = cell._replace(food=0)
            state = State(cell=cell, neighbors=neighbors)

        return state, reward, False

    # TODO: Should accept grid for custom maps?
    def reset(self):
        '''
        Reset environment to the initial state.
        Returns the initial state.
        '''
        self.grid = self.grid_cls()
        self.agent_pos = 0, 0
        state = State(cell=self.grid[self.agent_pos],
                      neighbors=self.grid.neighbors(*self.agent_pos))
        return state

    # All actions
    _all_actions = None

    @staticmethod
    def all_actions():
        ''' Returns the list of all possible actions. '''
        if Env._all_actions is None:
            Env._all_actions = [
                Action(ActionType.MOVE, di) for di in range(6)]
        return Env._all_actions


if __name__ == '__main__':
    env = Env()
    state = env.reset()
    done = False
    tot_reward = 0
    while not done:
        action = np.random.choice(env.all_actions())
        state, reward, done = env.step(action)
        tot_reward += reward
        print(state, reward, tot_reward)
