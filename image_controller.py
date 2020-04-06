import kivy
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ListProperty, NumericProperty
import numpy as np

class ImageController(BoxLayout):

    cursor_pos = ListProperty([0, 0])
    cursor_size = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super(ImageController, self).__init__(**kwargs)

    def update_palm(self, hand_pos):
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

