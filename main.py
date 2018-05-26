import numpy as np
from vispy import scene, visuals, app
from vispy.util import ptime
import scipy.integrate as spi
from time import sleep

import SimpleOscillatorVisualization
import SimpleOscillator


# Create a canvas to display our visual
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 800
ARENA_WIDTH = 800
ARENA_HEIGHT = WINDOW_HEIGHT
DATA_WIDTH = WINDOW_WIDTH - ARENA_WIDTH
DATA_HEIGHT = WINDOW_HEIGHT
ARENA_CENTER = (ARENA_WIDTH/2., ARENA_HEIGHT/2.)
ARENA_EDGE_SIZE = 100.0


def xy_location_to_pixel_location(x, y):
    global ARENA_WIDTH, ARENA_HEIGHT, ARENA_EDGE_SIZE
    px, py = x*ARENA_WIDTH/ARENA_EDGE_SIZE + ARENA_WIDTH/2., -1*y*ARENA_HEIGHT/ARENA_EDGE_SIZE + ARENA_HEIGHT/2.
    # print "{},{}  -->  {},{}".format(x, y, px, py)
    return px, py

# remember 0, 0 is upper left in pixel coordinates, (pixel_width, pixel_height) is the lower right in pixel coordinates
# In real coordinates 0, 0 is the center, negatives are to the left and down
CANVAS = scene.SceneCanvas(keys='interactive', show=True, size=(WINDOW_WIDTH, WINDOW_HEIGHT))
ARENA_VIEW = scene.widgets.ViewBox(parent=CANVAS.scene, name="arena_view", margin=0, bgcolor=(1, 1, 1, 1), size=(ARENA_WIDTH, ARENA_HEIGHT), pos=(0, 0))
DATA_VIEW = scene.widgets.ViewBox(parent=CANVAS.scene, name="data_view", margin=0, bgcolor=(0.8, 0.8, 0.8, 1), size=(DATA_WIDTH, DATA_HEIGHT), pos=(ARENA_WIDTH, 0))

oscillator_node = scene.visuals.create_visual_node(SimpleOscillatorVisualization.SimpleOscillatorVisual)
text_node = scene.visuals.create_visual_node(visuals.TextVisual)

COLORS = {"pid": (0, .6, .6, 1),
          "black": (0, 0, 0, 1),
          "gray": (0.6, 0.6, 0.6, 1),
          "red": (1, 0, 0, 1),
          "green": (0, 0.7, 0.5, 1)}

OSCILLATOR_VISUALS = {"pid": oscillator_node(ARENA_CENTER[0], ARENA_CENTER[1], 20, COLORS["pid"], parent=CANVAS.scene), }

TEXT_BOXES = {"time": text_node("t = ", pos=(ARENA_WIDTH + 100, 30), parent=CANVAS.scene, bold=True, font_size=30),
              "pos": text_node("x = ", pos=(ARENA_WIDTH + 100, 70), parent=CANVAS.scene, bold=True, font_size=30),
              "force": text_node("f = ", pos=(ARENA_WIDTH + 100, 110), parent=CANVAS.scene, bold=True, font_size=30)}

LINES = {"goal": scene.visuals.Line(pos=np.zeros((2, 2), dtype=np.float32), color=COLORS["red"], parent=CANVAS.scene),
         "base": scene.visuals.Line(pos=np.array([[ARENA_CENTER[0], ARENA_CENTER[1]-ARENA_HEIGHT/2], [ARENA_CENTER[0], ARENA_CENTER[1] + ARENA_HEIGHT/2]], dtype=np.float32),
                                    color=COLORS["black"], parent=CANVAS.scene, width=50),
         }

for k in LINES:
    LINES[k].transform = scene.transforms.STTransform()

OSCILLATORS = {"pid": SimpleOscillator.SimpleOscillator(k=0.1, c=0.1, goal=100, g=0, max_force=100, control_hz=5)}

TRANSITION_EXPERIENCES = list()

ITERATION_INTERVAL = 0.1  # time in seconds that passes each interval
FIRST_TIME = 0
LAST_TIME = 0
TOTAL_ITERATIONS = 0


def iterate(event):  # event is unused
    global FIRST_TIME, LAST_TIME, CANVAS, TIME_DILATION, TOTAL_ITERATIONS, CANVAS, GLOBAL_TIMER

    GLOBAL_TIMER.stop()

    if TOTAL_ITERATIONS < 1:
        FIRST_TIME = ptime.time()  # there is a huge gap in time as the window opens, so we need this manual time reset for the very first iteration
    TOTAL_ITERATIONS += 1
    current_time = TOTAL_ITERATIONS*ITERATION_INTERVAL
    # print "Total iterations = {}, t = {}".format(TOTAL_ITERATIONS, current_time)
    TEXT_BOXES["time"].text = "t = {}".format(SimpleOscillatorVisualization.format_time_string(current_time, 2))
    times = np.linspace(LAST_TIME, current_time, 10)

    for k in OSCILLATORS:
        oscillator = OSCILLATORS[k]
        oscillator.getAction(current_time)
        states = spi.odeint(SimpleOscillator.simpleOscillatorODE, oscillator.getState(), times, (oscillator,))
        oscillator.setState(states[-1])
        OSCILLATOR_VISUALS[k].new_pose(ARENA_CENTER[0] + oscillator.getState()[0], ARENA_CENTER[1])
        TEXT_BOXES["pos"].text = "x = {:.2f}".format(oscillator.getState()[0])
        TEXT_BOXES["force"].text = "f = {:.2f}".format(oscillator.getLastAction())
        LINES["goal"].set_data(pos=np.array([[ARENA_CENTER[0] + oscillator.getGoal(), ARENA_CENTER[1] - ARENA_HEIGHT/2],
                                             [ARENA_CENTER[0] + oscillator.getGoal(), ARENA_CENTER[1] + ARENA_HEIGHT/2]], dtype=np.float32))
        if np.abs(oscillator.getState()[0] - oscillator.getGoal()) < 1:
            if np.abs(oscillator.getState()[1]) < 1:
                TEXT_BOXES["pos"].color = COLORS["green"]
                print("Oscillator {} reached goal state in {} seconds".format(oscillator.getName(), current_time))
                GLOBAL_TIMER.disconnect(iterate)  # stop the simulation by disconnecting this callback
                return

    LAST_TIME = current_time
    CANVAS.update()

    GLOBAL_TIMER.start()
    return

GLOBAL_APP = app.Application()
GLOBAL_TIMER = app.Timer(interval='auto', connect=iterate, start=True)

if __name__ == "__main__":
    CANVAS.update()
    GLOBAL_APP.run()


