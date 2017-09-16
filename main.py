import kivy
# kivy requires this statement to be before other kivy imports
kivy.require('1.10.0')

from kivy.app import App

from kivy.graphics import Mesh, Color
from kivy.properties import ObjectProperty
from kivy.uix.scatter import ScatterPlane
from kivy.uix.widget import Widget

from collections import namedtuple
from itertools import chain
import numpy as np
import random


class Agent(Widget):
    '''
    A programmable agent to simulate individuals in the swarm.

    An Agent should act in accordance with some goal, as collecting the most
    food.
    '''
    cell = ObjectProperty(None)

    def on_cell(self, instance, new_cell):
        self.center = new_cell.center

    def act(self):
        '''
        Make a turn.
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
        self.cell = random.choice(self.cell.adj)


HexInfo = namedtuple(
    'HexInfo', ['x_off', 'y_off', 'size', 'mesh_mode', 'color'])


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
    def _pixcenter(self):
        ''' Returns pixel coordinates of the center. '''
        x = self.size * np.sqrt(3) * (self.q + self.r / 2)
        y = self.size * 3 / 2 * self.r
        return x, y

    @property
    def vertices(self):
        x, y = self._pixcenter
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


class Field(ScatterPlane):
    '''
    This is the Field which will contain cells.
    '''

    def __init__(self, cell_size=25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_size = cell_size
        self.hex_info = HexInfo(
            x_off=self.center_x,
            y_off=self.center_y,
            size=cell_size,
            mesh_mode='line_loop',
            color=Color(),
        )

        self.origin = Hex(0, 0, cell_size)
        self.grid = {(0, 0): self.origin}

        self.bind(pos=self.redraw, size=self.redraw)

    def redraw(self, *args):
        cx, cy = self.to_local(*self.center)
        self.canvas.clear()
        for _, h in self.grid.items():
            vertices = list(chain(*[(x + cx, y + cy, 0, 0)
                                    for x, y in h.vertices]))
            indices = list(range(6))
            self.canvas.add(self.hex_info.color)
            self.canvas.add(Mesh(vertices=vertices, indices=indices,
                                 mode=self.hex_info.mesh_mode))

    def on_touch_down(self, touch):
        super().on_touch_down(touch)

        x, y = self.to_local(touch.x, touch.y)
        cx, cy = self.to_local(*self.center)
        x -= cx
        y -= cy
        q, r = self.origin.pixel_to_hex(x, y)
        if (q, r) in self.grid:
            print("Touched ({}, {}) in {}.".format(q, r, (x, y)))
        else:
            new_cell = Hex(q, r, self.hex_info.size)
            self.grid[(q, r)] = new_cell
            for di in range(6):
                nei_c = Hex.neighbor_coordinates(q, r, di)
                if nei_c not in self.grid:
                    self.grid[nei_c] = new_cell.create_neighbor(di)
            self.redraw()

        return True


class SwarmApp(App):

    def update(self, dt):
        self.agent.act()
        self.agent.cell.create_adj_cells()

    def build(self):
        root = Field()

        # self.agent = Agent(cell=self.cell, size=[25, 25])
        # root.add_widget(self.agent)

        # Clock.schedule_interval(self.update, 1)

        return root


if __name__ == '__main__':
    SwarmApp().run()
