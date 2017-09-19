import kivy
# kivy requires this statement to be before other kivy imports
kivy.require('1.10.0')

from kivy.app import App

from kivy.clock import Clock
from kivy.graphics import Mesh, Color
from kivy.properties import ObjectProperty
from kivy.uix.scatter import ScatterPlane
from kivy.uix.widget import Widget

from collections import namedtuple
from itertools import chain
import numpy as np
import random
from enum import Enum


class AgentWidget(Widget):
    '''
    Widget responsible for drawing an agent.
    '''
    pass


# Inheriting namedtuple to add a docstring.
class State(namedtuple('State', ['cell', 'neighbors'])):
    '''
    State contains cell in which agent is situated and all adjacent cells.
    '''
    pass


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

    def perform(self, grid, state):
        ''' Returns the new state in accordance with action. '''
        if self.act_type is ActionType.MOVE:
            cell = state.neighbors[self.di]
            neighbors = grid.neighbors(cell)
            return State(cell=cell, neighbors=neighbors)
        return state


class Agent(object):
    '''
    A programmable agent to simulate individuals in the swarm.

    An Agent should act in accordance with some goal, as collecting the most
    food.
    '''
    def __init__(self, all_actions):
        ''' Initiates agent. '''
        self.all_actions = all_actions

    def act(self, state):
        '''
        Make a turn. Returns an action that agent wants to perform.
        state should convey information about location and neighbors.

        All agents move at the same time.
        During the turn they can:
        1) read information from the cell they are in (maybe, also from
           adjacent cells)
        2) put information into the cell they are in (maybe, also adjacent
           cell)
        3) move to the adjacent cell
        4) pick up food from cell
        5) put food into cell
        The information they read and put should be simple.
        '''
        pass


# TODO: Don't need this class. can add tuples in a separate method or generate
# neighbors, as well as generate the list of vertices in the Field class.
# Unless they'll contain some information.
class Hex(object):
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    def __init__(self, q, r, size):
        '''
        Initiates the Hex, with the coordinates q, r in Axial coordinates
        and the origin in (x_off, y_off) in the pixel coordinates.

        For more information on the Axial coordinates and hex coordinate system
        and algorithms check out http://www.redblobgames.com/grids/hexagons/ by
        Amit Patel.
        '''
        self.q = q
        self.r = r
        self.size = size

    def create_neighbor(self, di):
        '''
        Returns newly created neighbor of the cell in the direction di,
        which should be between 0 and 6 as per Hex.dirs.
        '''
        if di < 0 or di > 6:
            raise NotImplemented
        return self + Hex.dirs[di]

    @staticmethod
    def neighbor_coordinates(q, r, di):
        '''
        Returns tuple of coordinates of the neighbor in the given direction di.
        '''
        if di < 0 or di > 6:
            raise NotImplemented
        return (Hex.dirs[di][0] + q, Hex.dirs[di][1] + r)

    @property
    def pixcenter(self):
        ''' Returns pixel coordinates of the center. '''
        x = self.size * np.sqrt(3) * (self.q + self.r / 2)
        y = self.size * 3 / 2 * self.r
        return float(x), float(y)

    @property
    def vertices(self):
        x, y = self.pixcenter
        return [(float(x + np.cos(np.pi * (i + 0.5) / 3) * self.size),
                 float(y + np.sin(np.pi * (i + 0.5) / 3) * self.size))
                for i in range(6)]

    def pixel_to_hex(self, x, y):
        '''
        Converts pos in pixel coordinates with offset to (q, r) pair in
        Axial coordinates.
        '''
        q = (x * np.sqrt(3) / 3 - y / 3) / self.size
        r = y * 2 / 3 / self.size

        return Hex.axial_round(q, r)

    @staticmethod
    def axial_round(q, r):
        '''
        Converting to Cube coordinates, rounding up to nearest integers and
        resetting the component with biggest change so that x+y+z=0 stands.
        '''
        ccs = [q, -q - r, r]
        ind = np.argmax([np.abs(np.rint(x) - x) for x in ccs])
        ccs = [int(np.rint(x)) for x in ccs]
        ccs[ind] = - ccs[(ind + 1) % 3] - ccs[(ind + 2) % 3]

        return ccs[0], ccs[2]

    def __add__(self, other):
        if isinstance(other, Hex):
            return Hex(self.q + other.q, self.r + other.r, self.size)
        elif isinstance(other, tuple) and len(other) == 2:
            return Hex(self.q + other[0], self.r + other[1], self.size)
        else:
            raise NotImplemented

    def __radd__(self, other):
        if isinstance(other, tuple):
            return Hex(self.q + other[0], self.r + other[1], self.size)
        else:
            raise NotImplemented

    def __iadd__(self, other):
        if isinstance(other, Hex):
            self.q += other.q
            self.r += other.r
            return self
        elif isinstance(other, tuple) and len(other) == 2:
            self.q += other[0]
            self.r += other[1]
            return self
        else:
            raise NotImplemented


# TODO: Still feels strange to have so much logic here and feels not right to
# inherit dict. Use agregation instead?
# Operations with canvas are very implicit and nothing is deleted. Well, it is
# assumed that that grid cells won't be removed, which makes sense, but what if
# some cells will have to be rendered differently. Can maybe draw on top and
# then remove?
class Grid(dict):

    def __init__(self, canvas, mesh_mode, color, cell_size, pix_origin=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.canvas = canvas
        self.mesh_mode = mesh_mode
        self.cell_size = cell_size
        self.pix_origin = pix_origin
        self.canvas.add(color)

    def __missing__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            return super().__missing__(self, key)
        cell = Hex(key[0], key[1], self.cell_size)
        # TODO: Very much not sure if missing should assign a new key to value
        # Can istead remap __getitem__ to return self.getdefaulr(key, cell)
        # after checking if the element is in the dict. It might be fine just
        # to return what could have been here, the problem that it won't be
        # added to canvas.
        # self[key] = cell
        return cell

    def __setitem__(self, key, cell):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError

        cx, cy = self.pix_origin
        vertices = list(chain(*[(x + cx, y + cy, 0, 0)
                                for x, y in cell.vertices]))
        indices = list(range(6))
        # TODO: Should this be here? Where should it be?
        self.canvas.add(
            Mesh(vertices=vertices, indices=indices, mode=self.mesh_mode))
        return super().__setitem__(key, cell)

    def set_default(self, key):
        ''' Sets the default value to the key. '''
        if key not in self:
            self[key] = self.__missing__(key)

    def neighbors(self, cell):
        result = []
        for di in range(6):
            nq, nr = Hex.neighbor_coordinates(cell.q, cell.r, di)
            result.append(self[(nq, nr)])
        return result


class Field(ScatterPlane):
    '''
    This is the Field which will contain cells.
    '''
    agent_widget = ObjectProperty(None)

    def __init__(self, cell_size=25, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.origin = Hex(0, 0, cell_size)
        self.grid = Grid(self.canvas, 'line_loop', Color(), cell_size)

        # At the __init__ height and width, and consecutively center may be not
        # established, yet due to layout logic.
        Clock.schedule_once(self._init_after)

        Clock.schedule_interval(self.update, 1)

    def _init_after(self, dt):
        ''' Perform initializations after the layout is finalized. '''
        self.grid.pix_origin = self.to_local(*self.center)
        self.grid[(0, 0)] = self.origin
        self.state = State(cell=self.origin,
                           neighbors=self.grid.neighbors(self.origin))
        self._place_agent(self.state.cell)

    def _place_agent(self, cell):
        # TODO: Can use vectors with add capability?
        self.agent_widget.center = tuple(
            map(lambda x, y: x + y, self.grid.pix_origin, cell.pixcenter))

    def on_touch_down(self, touch):
        super().on_touch_down(touch)

        x, y = self.to_local(touch.x, touch.y)
        x -= self.grid.pix_origin[0]
        y -= self.grid.pix_origin[1]
        # FIXME: Doesn't look right
        # TODO: Make grid return q, r from touch local position
        q, r = self.origin.pixel_to_hex(x, y)
        if (q, r) in self.grid:
            print("Touched ({}, {}) in {}.".format(q, r, (x, y)))
        else:
            self.grid.set_default((q, r))
            for di in range(6):
                nei_c = Hex.neighbor_coordinates(q, r, di)
                self.grid.set_default(nei_c)

        return True

    # TODO: Shouldn't this feel better in SwarmApp?
    def update(self, dt):
        action = Action(ActionType.MOVE, random.choice(range(6)))
        self.state = action.perform(self.grid, self.state)
        self._place_agent(self.state.cell)
        self.grid.set_default((self.state.cell.q, self.state.cell.r))


class SwarmApp(App):

    def build(self):
        root = Field()
        return root


if __name__ == '__main__':
    SwarmApp().run()
