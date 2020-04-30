from kivy.uix.widget import Widget
from kivy.properties import *
from kivy.graphics.instructions import InstructionGroup
from kivy.clock import Clock
from kivy.graphics import Fbo, Color, Rectangle, Ellipse, BindTexture, Mesh, Line, RenderContext
from kivy.core.window import Window
import numpy as np
import matplotlib.pyplot as plt

DISTANCE_PER_POINT = 2.0
MASK_INTERVAL = 2
LINE_WIDTH_WINDOW = 10

FADE_FACTOR = 2
BASE_FADE = 0.5

mask_shader = """
$HEADER$

vec4 blur13(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.411764705882353) * direction;
  vec2 off2 = vec2(3.2941176470588234) * direction;
  vec2 off3 = vec2(5.176470588235294) * direction;
  color += texture2D(image, uv) * 0.1964825501511404;
  color += texture2D(image, uv + (off1 / resolution)) * 0.2969069646728344;
  color += texture2D(image, uv - (off1 / resolution)) * 0.2969069646728344;
  color += texture2D(image, uv + (off2 / resolution)) * 0.09447039785044732;
  color += texture2D(image, uv - (off2 / resolution)) * 0.09447039785044732;
  color += texture2D(image, uv + (off3 / resolution)) * 0.010381362401148057;
  color += texture2D(image, uv - (off3 / resolution)) * 0.010381362401148057;
  return color;
}

vec4 blur5(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3333333333333333) * direction;
  color += texture2D(image, uv) * 0.29411764705882354;
  color += texture2D(image, uv + (off1 / resolution)) * 0.35294117647058826;
  color += texture2D(image, uv - (off1 / resolution)) * 0.35294117647058826;
  return color;
}

// New uniform that will receive texture at index 1
uniform sampler2D texture1;

void main(void) {

    // multiple current color with both texture (0 and 1).
    // currently, both will use exactly the same texture coordinates.
    gl_FragColor = frag_color * \
        texture2D(texture0, tex_coord0) * \
        vec4(1, 1, 1, texture2D(texture1, tex_coord0).x); // blur13(texture1, tex_coord0, vec2(1), vec2(1, 0)).g);
        //texture2D(texture1, tex_coord0).x;
}
"""

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
        self.canvas = RenderContext()
        self.canvas.shader.fs = mask_shader
        self.buffer_container = None
        # We'll update our glsl variables in a clock
        Clock.schedule_interval(self.update_glsl, 0)

    def update_glsl(self, *largs):
        # This is needed for the default vertex shader.
        self.canvas['projection_mat'] = Window.render_context['projection_mat']
        self.canvas['modelview_mat'] = Window.render_context['modelview_mat']

    def on_size(self, instance, value):
        self.canvas.clear()
        with self.canvas:
            self.fbo = Fbo(size=value)
            self.mask_fbo = Fbo(size=(value[0] // 5, value[1] // 5), clear_color=(1, 1, 1, 1))
            Color(0, 1, 1, 0.9)
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
        dist = np.linalg.norm(np.array([last_point[0], last_point[1]]) -
                              np.array([old_point[0], old_point[1]]))
        depth_factor = 1 / (1 + np.exp(-(depth - 0.5) * 4))
        return np.clip((max_dist - dist) * (max_width - min_width) / (max_dist - min_dist) *
                       depth_factor,
                       min_width, max_width)

    def add_point(self, point, depth):
        """
        point: a point in window space to add to the paintbrush trajectory (x, y).
        depth: a 0-1 value indicating the depth into the screen of the current point.

        Returns the current line width.
        """
        point = (point[0] - self.pos[0], point[1] - self.pos[1])
        self.points.append(point)

        # Build a segment of line
        line_width = 0
        if len(self.points) > 2:
            line_width = self.current_line_width(depth)
            old_line_width = (sum(self.line_widths) / len(self.line_widths)
                              if self.line_widths else line_width)
            self.line_widths.append(line_width)
            if len(self.line_widths) > LINE_WIDTH_WINDOW:
                self.line_widths.pop(0)
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
                with self.mask_fbo:
                    self.mask_fbo.clear()
                    self.mask_fbo.clear_buffer()
                    white_values = 1 / (1 + np.exp(-((np.arange(len(self.mask_lines)) -
                                                     len(self.mask_lines)) / FADE_FACTOR + 3)))
                    white_values = white_values * (1 - BASE_FADE) + BASE_FADE
                    for white, line in zip(white_values, self.mask_lines):
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
        with self.fbo:
            self.fbo.clear_buffer()
        self.mask_lines = []
        with self.mask_fbo:
            self.mask_fbo.clear()
            self.mask_fbo.clear_buffer()
        self.on_size(self, self.size)
