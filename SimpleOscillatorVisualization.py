import numpy as np
from vispy import app, gloo, visuals, scene
# from vispy.visuals.transforms import *
from vispy.util import ptime

# Define a simple vertex shader. Template position and transform.
vertex_shader = """
void main() {
   gl_Position = $transform(vec4($position, 0, 1));
}
"""

# Very simple fragment shader. Template color.
fragment_shader = """
void main() {
  gl_FragColor = $color;
}
"""


def format_time_string(time, decimals):
    time = np.round(time, decimals)
    if decimals == 0:
        return "{}".format(int(time))
    fraction = time - np.floor(time)
    fraction_as_integer = "{}".format(int(np.power(10, decimals)*fraction))
    return "{}.{}".format(int(time), fraction_as_integer)


class TimeTextVisual(visuals.TextVisual):
    """
    Display the string "t = X" where X is the time in seconds.
    """
    def __init__(self, text, color='black', bold=False, italic=False, face='OpenSans', font_size=12, pos=[0, 0, 0], rotation=0.0, anchor_x='center', anchor_y='center', font_manager=None):
        super(TimeTextVisual, self).__init__(text, color, bold, italic, face, font_size, pos, rotation, anchor_x, anchor_y, font_manager)
        self.unfreeze()  # super class froze things. Need to unfreeze.
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)
        self.timer = app.Timer(interval='auto', connect=self.update_time, start=False)
        self._time = 0.0
        self._first_time = ptime.time()
        self._last_time = ptime.time()
        self.text = text
        self.freeze()
        self.shared_program['time'] = self._time
        self.shared_program['text_scale'] = 1
        self.timer.start()

    def update_time(self, ev):  # argument ev is required for scene, but doesn't have to be used
        t = ptime.time()
        self._time += t - self._last_time
        self._last_time = t
        self.shared_program['time'] = self._time
        x = t - self._first_time
        self.text = "t = {%.4f}".format(x)
        self.update()


class SimpleOscillatorVisual(visuals.Visual):
    """
    Parameters
    ----------
    x : float
        x coordinate of origin
    y : float
        y coordinate of origin
    w : float
        width of max box
    rgba:
        color of the mass box

    origin is in the center 1/2 width, 1/2 length
    basic orientation is to the right -->
    """
    def __init__(self, x, y, w, rgba):
        # Initialize the visual with a vertex shader and fragment shader
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)
        self._x = x
        self._y = y
        self._w = w

        self.vbo = gloo.VertexBuffer(np.array([
            [x-0.5*w, y+0.5*w],
            [x-0.5*w, y-0.5*w],
            [x+0.5*w, y-0.5*w],
            [x+0.5*w, y+0.5*w]
        ], dtype=np.float32))

        # Assign values to the $position and $color template variables in the shaders.
        self._draw_mode = 'line_loop'
        self.freeze()  # no more attributes
        self.shared_program.vert['position'] = self.vbo
        self.shared_program.frag['color'] = tuple(rgba)

    def _prepare_transforms(self, view):
        view.view_program.vert['transform'] = view.get_transform()

    def new_pose(self, x, y):
        self._x = x
        self._y = y
        w = self._w
        vertices = np.array(
            [
                [x - 0.5 * w, y + 0.5 * w],
                [x - 0.5 * w, y - 0.5 * w],
                [x + 0.5 * w, y - 0.5 * w],
                [x + 0.5 * w, y + 0.5 * w]
            ], dtype=np.float32)
        self.vbo = gloo.VertexBuffer(vertices)
        self.shared_program.vert['position'] = self.vbo

if __name__ == '__main__':
    app.run()