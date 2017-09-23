import kivy
# kivy requires this statement to be before other kivy imports
kivy.require('1.10.0')

from kivy.app import App

from kivy.clock import Clock
from kivy.graphics import Mesh, Color
from kivy.graphics.instructions import InstructionGroup
from kivy.properties import ObjectProperty, NumericProperty
from kivy.uix.scatter import ScatterPlane
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout

from itertools import chain
import numpy as np
import random

from env import BaseGrid, Env


class AgentWidget(Widget):
    '''
    Widget responsible for drawing an agent.
    '''
    pass


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


class Grid(BaseGrid):

    def __init__(self, canvas, mesh_mode, color, cell_size, pix_origin=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.canvas = canvas
        self.mesh_mode = mesh_mode
        self.cell_size = cell_size
        self.pix_origin = pix_origin
        self.canvas_groups = {}
        self.color = color

    def _clean_key(self, key):
        super()._clean_key(key)
        self.canvas.remove(self.canvas_groups[key])

    def __setitem__(self, key, cell):
        ret = super().__setitem__(key, cell)

        group = InstructionGroup()
        if cell.food == 0:
            group.add(self.color)
            group.add(Mesh(vertices=self._mesh_vertices(cell),
                           indices=list(range(6)),
                           mode=self.mesh_mode))
        else:
            group.add(Color(0, 1, 0, 1))
            group.add(Mesh(vertices=self._mesh_vertices(cell),
                           indices=list(range(6)),
                           mode='triangle_fan'))

        self.canvas_groups[key] = group
        self.canvas.add(group)
        return ret

    def _mesh_vertices(self, cell):
        x, y = self.pixcenter(cell.q, cell.r)
        return list(chain(*[
            (float(x + np.cos(np.pi * (i + 0.5) / 3) * self.cell_size),
             float(y + np.sin(np.pi * (i + 0.5) / 3) * self.cell_size),
             0, 0) for i in range(6)]))

    def pixel_to_hex(self, x, y):
        '''
        Converts pos in pixel coordinates with offset to (q, r) pair in
        Axial coordinates.
        '''
        x -= self.pix_origin[0]
        y -= self.pix_origin[1]
        q = (x * np.sqrt(3) / 3 - y / 3) / self.cell_size
        r = y * 2 / 3 / self.cell_size

        return Grid.axial_round(q, r)

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

    def pixcenter(self, q, r):
        ''' Returns pixel coordinates of the center. '''
        cx, cy = self.pix_origin
        x = cx + self.cell_size * np.sqrt(3) * (q + r / 2)
        y = cy + self.cell_size * 3 / 2 * r
        return float(x), float(y)


class Field(ScatterPlane):
    '''
    This is the Field which will contain cells.
    '''
    agent_widget = ObjectProperty(None)
    total_reward = NumericProperty(0)

    def __init__(self, cell_size=25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_size = cell_size

        # At the __init__ height and width, and consecutively center may be not
        # established, yet due to layout logic.
        Clock.schedule_once(self._init_after)

        Clock.schedule_interval(self.update, 1)

    def _init_after(self, dt):
        ''' Perform initializations after the layout is finalized. '''
        self.env = Env()
        self.grid = Grid(self.canvas, 'line_loop', Color(), self.cell_size,
                         self.to_local(*self.center))
        self.state = self.env.reset(self.grid)
        self._place_agent(self.state.cell)

    def _place_agent(self, cell):
        self.agent_widget.center = self.grid.pixcenter(cell.q, cell.r)
        for _ in self.grid.neighbors(cell.q, cell.r):
            pass

    def on_touch_down(self, touch):
        super().on_touch_down(touch)

        x, y = self.to_local(touch.x, touch.y)
        q, r = self.grid.pixel_to_hex(x, y)

        if (q, r) in self.grid:
            print("Touched ({}, {}) in {}.".format(q, r, (x, y)))
        else:
            self.grid.init(q, r)
            for _ in self.grid.neighbors(q, r):
                pass

        return True

    # TODO: Shouldn't this feel better in SwarmApp?
    def update(self, dt):
        action = random.choice(self.env.all_actions())
        self.state, reward, done = self.env.step(action)
        self.total_reward += int(reward)
        self._place_agent(self.state.cell)


class SwarmApp(App):

    def build(self):
        return BoxLayout()


if __name__ == '__main__':
    SwarmApp().run()
