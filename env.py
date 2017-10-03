# TODO: some small unit tests maybe?
import numpy as np
from collections import namedtuple, defaultdict
from enum import Enum


# TODO: Is better way to call it observation?
State = namedtuple('State', ['cell', 'neighbors', 'food'])
State.__doc__ += '''
State contains cell in which agent is situated and all adjacent cells.
'''


Hex = namedtuple('Hex', ['q', 'r', 'food', 'scent'])
Hex.__doc__ += '''
Contains information about the hex cell, like present food or trails.
Scent represents numerically how long ago someone passed here. The bigger the
value the more recently someone passed.
'''


class ActionType(Enum):
    ''' All possible action types the agent can take. '''
    MOVE = 1


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
        self._cells = {}

    def init(self, q, r):
        ''' Initiates cell with coordinates q, r. '''
        if (q, r) not in self:
            food = np.random.choice([1, 0], p=[0.3, 0.7])
            self.__setitem__((q, r), Hex(q, r, food, 0))

    def neighbor_dir(self, q, r, d):
        # This raise condition is not expected to be handled.
        assert 0 <= d < 6

        q += BaseGrid.dirs[d][0]
        r += BaseGrid.dirs[d][1]
        if (q, r) not in self._cells:
            self.init(q, r)
        return self._cells[q, r]

    def neighbors(self, q, r):
        return [self.neighbor_dir(q, r, d)
                for d, _ in enumerate(BaseGrid.dirs)]

    def __contains__(self, key):
        return key in self._cells

    def _clean_key(self, key):
        ''' Can clean up everything related to key. '''
        pass

    def __setitem__(self, key, val):
        if len(key) != 2:
            raise IndexError
        if key in self._cells:
            # TODO: Is it better to use __delitem__ ?
            self._clean_key(key)
        self._cells[key] = val

    def __getitem__(self, key):
        if len(key) != 2:
            raise IndexError
        if key not in self._cells:
            self.init(*key)
        return self._cells[key]


class Env(object):
    ''' Environment with which the agent can interact. '''
    def __init__(self, num_steps=None):
        self.grid = None
        self.num_steps = num_steps

    # TODO: break into simpler procedures
    def step(self, action):
        '''
        Performs the action and returns new state, reward and whether the
        state is terminal.
        '''
        reward = 0
        if action.act_type is ActionType.MOVE:
            cell = self.grid.neighbor_dir(
                self.agent_pos[0], self.agent_pos[1], action.di)
            neighbors = self.grid.neighbors(cell.q, cell.r)

            self.agent_pos = cell.q, cell.r

            if cell.food > 0 and self.agent_food == 0:
                self.agent_food += cell.food
                self.grid[cell.q, cell.r] = cell._replace(food=0)
                cell = self.grid[cell.q, cell.r]

            if self.agent_pos == self.home_pos and self.agent_food > 0:
                reward = self.agent_food
                self.agent_food = 0

            for i, ncell in enumerate(neighbors):
                if self.tvisited[ncell.q, ncell.r] < 0:
                    neighbors[i] = ncell._replace(scent=0)
                elif self.steps - self.tvisited[ncell.q, ncell.r] <= 1:
                    neighbors[i] = ncell._replace(scent=2)
                else:
                    neighbors[i] = ncell._replace(scent=1)

            # TODO: remove this cheat needed for easier sarsa feature
            #       engineering and finding a way back. Use more features and
            #       real valued state after implementing Q(state, action)
            #       approximation instead of table.
            if self.agent_food > 0:
                self.tvisited[cell.q, cell.r] = -1
            else:
                self.tvisited[cell.q, cell.r] = self.steps

            state = State(cell=cell, neighbors=neighbors, food=self.agent_food)

        self.steps += 1
        if self.num_steps is None:
            done = False
        else:
            done = self.steps >= self.num_steps
        return state, reward, done

    # TODO: Move num_steps to init
    def reset(self, grid=None):
        '''
        Reset environment to the initial state.
        Returns the initial state.
        '''
        if grid is None:
            grid = BaseGrid()
        self.grid = grid
        # Time of last visit into the cell. -1 standing for not visited.
        self.tvisited = defaultdict(lambda: -1)
        self.steps = 1
        # TODO: Move all the agent information into State?
        #       Clearer here and easier to generalise on more agents.
        self.agent_pos = 0, 0
        self.tvisited[self.agent_pos] = self.steps - 1
        self.agent_food = 0
        self.home_pos = 0, 0
        state = State(cell=self.grid[self.agent_pos],
                      neighbors=self.grid.neighbors(*self.agent_pos),
                      food=self.agent_food)
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
    env = Env(3)
    grid = BaseGrid()
    state = env.reset(grid)
    done = False
    tot_reward = 0
    while not done:
        action = np.random.choice(env.all_actions())
        state, reward, done = env.step(action)
        tot_reward += reward
        print(state, reward, tot_reward)
