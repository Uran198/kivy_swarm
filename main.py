import kivy
# kivy requires this statement to be before other kivy imports
kivy.require('1.10.0')

# flake8: noqa E402
from kivy.app import App
from kivy.graphics import Mesh
from kivy.properties import ListProperty, ObjectProperty
from kivy.uix.scatter import ScatterPlane
from kivy.uix.widget import Widget

from collections import deque
import numpy as np


def aa_sign(p1, p2, p3):
    ''' Returns acute angle sign of triple p1, p2, p3. '''
    return np.sign(np.cross(np.subtract(p3, p1), np.subtract(p2, p1)))


def all_cells(start):
    ''' Runs BFS and yields all the cells on the way. '''
    saw = {start}
    queue = deque([start])
    # TODO: Is len O(1) for deque?
    while len(queue) > 0:
        u = queue.pop()
        yield u
        for cell in u.adj:
            if cell not in saw:
                saw.add(cell)
                queue.append(cell)


# TODO: Create an object to keep track of all the agents, update and create
#       them.
class Agent(object):
    '''
    A programmable agent to simulate individuals in the swarm.

    An Agent should act in accordance with some goal, as collecting the most
    food.
    '''
    def __init__(self, cell_location=None):
        self.loc = cell_location

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
        pass


class Cell(Widget):
    '''
    Hexagon cell.
    NOTE: Actual width is a bit smaller from the saved width for the sake of
    simplicity. actual_width = sqrt(3)/2 * saved_width
    '''
    vertices = ListProperty(None)
    adj = ListProperty([])

    def collide_point(self, x, y):
        if super().collide_point(x, y):
            xy = [(x, y) for x, y, _, _ in self.vertices]
            cent = tuple(self.center)
            for p1, p2 in zip(xy, xy[1:] + xy[:1]):
                if aa_sign(p1, p2, cent) * aa_sign(p1, p2, (x, y)) < 0:
                    return False
            return True
        return False

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            print("Touched {}".format(self.center))
            self.create_adj_cells()
            # return True
        return super().on_touch_down(touch)

    def create_adj_cells(self, center=None, size=None):
        '''
        Creates adjacent cells if they are not already created.
        center and size arguments needed when creating cells before all the
        intiation of width and center are done. And that of the parent widget.
        '''
        if not center:
            center = self.center
        if not size:
            size = self.size
        r = self.width * np.sqrt(3)/2
        print(self.center)
        print(self.width)
        print(len(self.adj))
        for i in range(6):
            angle = self.hex_step*(i+0.5)
            x = float(self.center_x + np.cos(angle)*r)
            y = float(self.center_y + np.sin(angle)*r)
            if not any(cell.collide_point(x, y) for cell in self.adj):
                new_cell = Cell(prev=self, angle=angle, size=self.size)
                for cell in [c for c in all_cells(self)
                             if np.linalg.norm(
                             np.subtract(c.center, new_cell.center)) < r*1.5]:
                        cell.adj.append(new_cell)
                        new_cell.adj.append(cell)
                self.parent.add_widget(new_cell)
        print(len(self.adj))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Field(ScatterPlane):
    '''
    This is the Field which will contain cells.
    '''
    pass


class SwarmApp(App):

    def build(self):
        root = Field()
        cell_size = 50, 50

        self.cell = Cell(size=cell_size, center=[200, 200])
        root.add_widget(self.cell)

        self.cell.create_adj_cells()
        self.cell.size = 60, 60
        self.cell.pos = [500, 500]

        return root


if __name__ == '__main__':
    SwarmApp().run()
