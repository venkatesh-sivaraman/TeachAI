import kivy
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import *
from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Ellipse, Color
import numpy as np
from point_util import *
from region_detector import *

class ImageContainer(FloatLayout):
    image_source = StringProperty('test_image.jpg')

class GestureDrawing(Widget):

    points = ListProperty([])

    def __init__(self, **kwargs):
        super(GestureDrawing, self).__init__(**kwargs)

    def add_point(self, point):
        for elem in point:
            self.points.append(elem)

CURSOR_TRAIL_LENGTH = 20
CURSOR_RGB = (0.98, 0.85, 0.16)
CURSOR_ON_RGB = (0.23, 0.9, 0.99)
CURSOR_ELLIPSE_SIZE = 30.0

class CursorTrail(InstructionGroup):
    points = []
    colors = []
    on_colors = []

    def __init__(self, **kwargs):
        super(CursorTrail, self).__init__(**kwargs)
        # Create a trail of colors
        for opacity in np.linspace(0, 1, CURSOR_TRAIL_LENGTH):
            self.colors.append(Color(*CURSOR_RGB, opacity))
            self.on_colors.append(Color(*CURSOR_ON_RGB, opacity))

    def update_instructions(self):
        self.clear()
        for i, (point, active) in enumerate(self.points):
            self.add(self.colors[i] if not active else self.on_colors[i])
            self.add(point)

    def add_point(self, point, radius, is_active):
        self.points.append((Ellipse(pos=(point[0] - radius,
                                        point[1] - radius),
                                   size=(radius * 2, radius * 2)), is_active))
        if len(self.points) > CURSOR_TRAIL_LENGTH:
            self.points.pop(0)
        self.update_instructions()

    def pop_point(self):
        if not self.points:
            return
        self.points.pop(0)
        self.update_instructions()

class ImageController(FloatLayout):

    cursor_pos = ListProperty([0, 0])
    cursor_size = NumericProperty(0.0)
    segmentation = ObjectProperty(None)
    image_src = StringProperty("test_image.jpg")

    def __init__(self, **kwargs):
        super(ImageController, self).__init__(**kwargs)
        self.gesture_points = []
        self.cursor_trail = None
        self.ids.speech_widget.bind(transcript=self.speech_transcript_completed)

    def reset(self):
        self.segmentation = None
        self.ids.speech_widget.reset()

    def update(self, dt):
        self.ids.speech_widget.update(dt)
        if not self.ids.speech_widget.is_recording:
            self.ids.speech_widget.start_recording()

    def on_image_src(self, instance, value):
        self.ids.image_view.image_source = value

    def update_palm(self, hand_pos):
        if hand_pos is None:
            if self.cursor_trail:
                self.cursor_trail.pop_point()
            return

        pos = np.array(hand_pos[:2])
        leap_corner = np.array([-100.0, 100.0])
        leap_size = np.array([200.0, 200.0])
        screen_bounds = np.array(self.size)
        self.cursor_pos = ((pos - leap_corner) * screen_bounds / leap_size).tolist()

        leap_z = -100.0
        leap_z_size = 200.0
        cursor_max = 40.0
        new_size = (hand_pos[2] - leap_z) * cursor_max / leap_z_size
        self.cursor_size = float(np.clip(new_size, 0, cursor_max))

        image_pos = convert_point(self.cursor_pos, self, self.ids.image_view)
        if self.gesture_points:
            # Add point to existing region
            self.gesture_points.append(image_pos)
        elif new_size <= cursor_max * 0.5 and not self.gesture_points:
            self.gesture_points.append(image_pos)

        if not self.cursor_trail:
            self.cursor_trail = CursorTrail()
            self.ids.image_view.canvas.add(self.cursor_trail)
        self.cursor_trail.add_point(image_pos, cursor_max - new_size, len(self.gesture_points) > 0)

    def speech_transcript_completed(self, instance, value):
        if not value:
            return
        if self.gesture_points:
            region = detect_region(self.gesture_points,
                                   (int(self.ids.image_view.size[0]),
                                    int(self.ids.image_view.size[1])))
            region.save_image(CURSOR_RGB)
        else:
            region = None
        self.gesture_points = []
        self.segmentation = {'speech': value, 'gesture': region}
