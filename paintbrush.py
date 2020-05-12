from kivy.uix.widget import Widget
from kivy.properties import *
from kivy.graphics.instructions import InstructionGroup
from kivy.clock import Clock
from kivy.graphics import Fbo, Color, Rectangle, Ellipse, BindTexture, Mesh, Line, RenderContext
from kivy.core.window import Window
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

DISTANCE_PER_POINT = 2.0
MASK_INTERVAL = 2
LINE_WIDTH_WINDOW = 10

FADE_FACTOR = 2
BASE_FADE = 0.2

AUTOCORRELATION_WINDOW = 128

mask_shader = """
$HEADER$

// New uniform that will receive texture at index 1
uniform sampler2D texture1;

void main(void) {

    // multiple current color with both texture (0 and 1).
    // currently, both will use exactly the same texture coordinates.
    gl_FragColor = frag_color * \
        texture2D(texture0, tex_coord0) * \
        vec4(1, 1, 1, texture2D(texture1, tex_coord0).x);
}
"""

# 0 = inverse speed
# 1 = depth
# 2 = periodicity + depth
PAINTBRUSH_MODE = 0

class Paintbrush(Widget):
    def __init__(self, **kwargs):
        super(Paintbrush, self).__init__(**kwargs)
        self.fbo = Fbo(size=(10, 10))
        self.mesh = Mesh()
        self.points = []
        self.vertices = []
        self.indices = []
        self.line_widths = []
        self.cap_vertices_index = 0
        self.cap_indices_index = 0
        self.mask_lines = []
        self.mask_alphas = []
        self.canvas = RenderContext()
        self.canvas.shader.fs = mask_shader
        self.buffer_container = None
        self.rgb = (0, 1, 1)
        # We'll update our glsl variables in a clock
        Clock.schedule_interval(self.update_glsl, 0)

        # Maintain a window of history for autocorrelations
        self.ac_window = []
        self.ac_position = 0
        self.periodicity_factor = 1.0

    def update_glsl(self, *largs):
        # This is needed for the default vertex shader.
        self.canvas['projection_mat'] = Window.render_context['projection_mat']
        self.canvas['modelview_mat'] = Window.render_context['modelview_mat']

    def on_size(self, instance, value):
        self.canvas.clear()
        with self.canvas:
            self.fbo = Fbo(size=value)
            self.mask_fbo = Fbo(size=(value[0] // 5, value[1] // 5), clear_color=(1, 1, 1, 1))
            Color(*self.rgb, 0.9)
            BindTexture(texture=self.mask_fbo.texture, index=1)
            self.buffer_container = Rectangle(pos=self.pos, size=value, texture=self.fbo.texture)
            #Rectangle(pos=self.pos, size=value, texture=self.mask_fbo.texture)

        self.canvas['texture1'] = 1

        with self.fbo:
            Color(1, 1, 1)
            self.mesh = Mesh(mode='triangle_strip')

    def on_pos(self, instance, value):
        if not self.buffer_container: return
        self.buffer_container.pos = value

    def build_line_segment(self, start, end, future, start_width=8.0, end_width=8.0):
        """Builds a line segment knowing the start and end, as well as one point in the future."""
        start = np.array([start[0], start[1]])
        end = np.array([end[0], end[1]])
        future = np.array([future[0], future[1]])
        length = np.linalg.norm(end - start)
        num_interpolants = max(int(length / DISTANCE_PER_POINT), 3)

        normal = (end - start) / length * start_width / 2.0
        normal = np.array([-normal[1], normal[0]])
        end_normal = (future - end) / max(np.linalg.norm(future - end), 0.1) * end_width / 2.0
        end_normal = np.array([-end_normal[1], end_normal[0]])
        delta_sign = None

        # if (self.last_normal is not None and
        #     np.linalg.norm(normal - self.last_normal) > np.linalg.norm(normal + self.last_normal)):
        #     self.last_normal *= -1

        # Add points deviating in alternating directions around the actual path
        for i in range(num_interpolants):
            path_point = start + (i / num_interpolants) * (end - start)
            delta = normal + (i / num_interpolants) * (end_normal - normal)
            if delta_sign is None:
                delta_sign = 1
                if len(self.points) > 3:
                    second_last_vertex = np.array(self.vertices[-8:-6])
                    option_1 = path_point + delta
                    option_2 = path_point - delta
                    if (np.linalg.norm(option_2 - second_last_vertex) <
                        np.linalg.norm(option_1 - second_last_vertex)):
                        delta_sign *= -1
            self.vertices.extend([*(path_point + delta * delta_sign), 0, 0])
            self.indices.append(len(self.indices))
            delta_sign *= -1

    def add_cap(self, width):
        """Adds a round line cap to the end of the vertex/index list."""
        self.cap_vertices_index = len(self.vertices)
        self.cap_indices_index = len(self.indices)
        if len(self.points) < 3:
            return

        # Extend the current line segment using a circular interpolation of line widths
        start = np.array([self.points[-1][0], self.points[-1][1]])
        prev = np.array([self.points[-2][0], self.points[-2][1]])
        end = start + (start - prev) / max(np.linalg.norm(start - prev), 0.001) * width / 2.0
        length = np.linalg.norm(end - start)
        num_interpolants = max(int(length / DISTANCE_PER_POINT) * 2, 3)

        normal = (end - start) / length * width / 2.0
        normal = np.array([-normal[1], normal[0]])
        end_normal = np.zeros(2)
        delta_sign = None

        # Add points deviating in alternating directions around the actual path
        for i in range(num_interpolants):
            path_point = start + (i / (num_interpolants - 1)) * (end - start)
            circ_progress = 1 - np.sqrt(1 - (i / (num_interpolants - 1)) ** 2)
            delta = normal + circ_progress * (end_normal - normal)
            if delta_sign is None:
                delta_sign = 1
                if len(self.points) > 3:
                    second_last_vertex = np.array(self.vertices[-8:-6])
                    option_1 = path_point + delta
                    option_2 = path_point - delta
                    if (np.linalg.norm(option_2 - second_last_vertex) <
                        np.linalg.norm(option_1 - second_last_vertex)):
                        delta_sign *= -1
            self.vertices.extend([*(path_point + delta * delta_sign), 0, 0])
            self.indices.append(len(self.indices))
            delta_sign *= -1

    def remove_cap(self):
        """Removes a cap on the line."""
        if self.cap_vertices_index > 0 and self.cap_vertices_index <= len(self.vertices):
            del self.vertices[self.cap_vertices_index:]
            del self.indices[self.cap_indices_index:]
        self.cap_vertices_index = 0
        self.cap_indices_index = 0

    def current_line_width(self, depth, window=5):
        """Computes the current line width of the previous `window` points."""
        max_width = 120.0
        min_width = 5.0
        min_dist = 40.0
        max_dist = 140.0
        last_point = self.points[-1]
        old_point = self.points[max(0, len(self.points) - window)]
        if PAINTBRUSH_MODE == 0:
            dist = np.linalg.norm(np.array([last_point[0], last_point[1]]) -
                                  np.array([old_point[0], old_point[1]]))
        else:
            dist = 120.0
        width = (max_dist - dist) * (max_width * 0.8 - min_width) / (max_dist - min_dist)
        if PAINTBRUSH_MODE != 0:
            depth_factor = 1 / (1 + np.exp(-(depth - 0.5) * 4))
            width *= depth_factor
            if PAINTBRUSH_MODE == 2:
                width *= self.periodicity_factor
        return np.clip(width, min_width, max_width)

    def update_periodicity(self, point):
        """Computes a new autocorrelation magnitude by adding the given point."""
        self.ac_window.append(point)
        if len(self.ac_window) > AUTOCORRELATION_WINDOW:
            del self.ac_window[0]
        self.ac_position += 1
        if self.ac_position % 8 == 0 and len(self.ac_window) == AUTOCORRELATION_WINDOW:
            ac_window = np.array(self.ac_window)
            x_fft = np.abs(np.fft.rfft(ac_window[:,0] * np.hanning(AUTOCORRELATION_WINDOW)))
            y_fft = np.abs(np.fft.rfft(ac_window[:,1] * np.hanning(AUTOCORRELATION_WINDOW)))
            x_fft = x_fft[4:20] / np.mean(x_fft[4:20])
            y_fft = y_fft[4:20] / np.mean(y_fft[4:20])
            # if self.ac_position > 200:
            #     plt.figure()
            #     plt.subplot(121)
            #     plt.plot(ac_window[:,0], ac_window[:,1])
            #     plt.subplot(122)
            #     plt.plot(x_fft, label='x')
            #     plt.plot(y_fft, label='y')
            #     plt.show()
            self.periodicity_factor = ((max(1.0, np.max(x_fft) / 4.0) *
                                        max(1.0, np.max(y_fft) / 4.0)) - 1) ** 2 + 1


    def add_point(self, point, depth=None, width=None, alpha=None):
        """
        point: a point in window space to add to the paintbrush trajectory (x, y).
        depth: a 0-1 value indicating the depth into the screen of the current point.
        alpha: a manual 0-1 alpha level for this point.

        Returns the current line width.
        """
        point = (point[0] - self.pos[0], point[1] - self.pos[1])
        self.points.append(point)

        # Build a segment of line
        line_width = 0
        if len(self.points) > 2:
            self.update_periodicity(point)
            line_width = self.current_line_width(depth) if depth is not None else width
            old_line_width = (sum(self.line_widths) / len(self.line_widths)
                              if self.line_widths else line_width)
            self.line_widths.append(line_width)
            if len(self.line_widths) > LINE_WIDTH_WINDOW:
                self.line_widths.pop(0)
            if width is None:
                line_width = sum(self.line_widths) / len(self.line_widths)
            # Clamp the amount by which the line width can change - results in
            # smoother lines
            # line_width = old_line_width + np.clip(line_width - old_line_width, -2.0, 2.0)
            self.remove_cap()
            self.build_line_segment(*self.points[-3:], old_line_width, line_width)
            self.add_cap(line_width)

            # Update mask
            if len(self.points) % MASK_INTERVAL == 0 and len(self.points) > MASK_INTERVAL:
                self.mask_lines.append(Line(points=(self.points[-MASK_INTERVAL - 1][0] / 5,
                                                    self.points[-MASK_INTERVAL - 1][1] /5 ,
                                                    self.points[-1][0] / 5,
                                                    self.points[-1][1] / 5),
                                           width=(line_width + 8.0) / 10))
                if alpha is not None:
                    self.mask_alphas.append(alpha)
                with self.mask_fbo:
                    self.mask_fbo.clear()
                    self.mask_fbo.clear_buffer()
                    if len(self.mask_alphas) == len(self.mask_lines):
                        white_values = self.mask_alphas
                    else:
                        white_values = 1 / (1 + np.exp(-((np.arange(len(self.mask_lines)) -
                                                         len(self.mask_lines)) / FADE_FACTOR + 3)))
                        white_values = white_values * (1 - BASE_FADE) + BASE_FADE
                    for i, (white, line) in enumerate(zip(white_values, self.mask_lines)):
                        Color(white, white, white, 1)
                        self.mask_fbo.add(line)


        # if len(self.points) % 100 == 20:
        #     plt.figure()
        #     plt.plot(self.vertices[::4], self.vertices[1::4])
        #     plt.plot(self.vertices[::4], self.vertices[1::4], 'b.')
        #     plt.plot([x[0] for x in self.points], [x[1] for x in self.points], 'ro')
        #     plt.plot([x[0] for x in self.points], [x[1] for x in self.points], 'r-')
        #     plt.show()

        # self.vertices.extend([point[0], point[1], 0, 0])
        # if len(self.points) > 1:
        #     self.indices.extend([len(self.points) - 2, len(self.points) - 1])
        self.mesh.vertices = self.vertices
        self.mesh.indices = self.indices
        return line_width

    def clear(self):
        self.points = []
        self.vertices = []
        self.indices = []
        self.mesh.vertices = []
        self.mesh.indices = []
        self.periodicity_factor = 1.0
        self.ac_window = []
        self.ac_position = 0
        with self.fbo:
            self.fbo.clear_buffer()
        self.mask_lines = []
        self.mask_colors = []
        with self.mask_fbo:
            self.mask_fbo.clear()
            self.mask_fbo.clear_buffer()
        self.on_size(self, self.size)
