from kivy.properties import *
from kivy.uix.widget import Widget
import numpy as np

HOVER_DURATION = 1.0
HOVER_TOLERANCE = 40.0
REFRACTORY_PERIOD = 3.0

class LeapButton(Widget):
    hover_color = ListProperty([0, 0, 1, 1])
    text = StringProperty("Button")

    def __init__(self, **kwargs):
        super(LeapButton, self).__init__(**kwargs)
        self.stationary_start_time = 0.0
        self.stationary_pos = None
        self.current_time = 0.0
        self.hold_time = 1.0
        self.on_click = None
        self.time_of_last_click = 0.0

    def update(self, dt):
        self.current_time += dt

    def set_hover_rgb(self, r, g, b):
        self.hover_color[:3] = [r, g, b]

    def update_palm(self, hand_pos, depth):
        if self.disabled:
            self.hover_color[-1] = 0.1
            self.stationary_pos = None
            self.stationary_start_time = 0.0
            return
        if hand_pos is None:
            self.hover_color[-1] = 0.2
            self.stationary_pos = hand_pos
            self.stationary_start_time = 0.0
            return
        hand_pos = (hand_pos[0] - self.pos[0], hand_pos[1] - self.pos[1])
        if (hand_pos[0] <= 0 or hand_pos[0] >= self.width or
            hand_pos[1] <= 0 or hand_pos[1] >= self.height):
            self.hover_color[-1] = 0.2
            self.stationary_pos = hand_pos
            self.stationary_start_time = 0.0
            return

        hand_pos = np.array(hand_pos)
        if (self.stationary_pos is not None and
            self.current_time - self.time_of_last_click >= REFRACTORY_PERIOD and
            np.linalg.norm(hand_pos - self.stationary_pos) <= HOVER_TOLERANCE):
            if self.stationary_start_time == 0.0:
                self.stationary_start_time = self.current_time
            elif self.current_time - self.stationary_start_time >= self.hold_time:
                if self.on_click:
                    self.on_click()
                    self.time_of_last_click = self.current_time
            else:
                time = self.current_time - self.stationary_start_time
                self.hover_color[-1] = max((time - self.hold_time * 0.2) / (self.hold_time * 0.8),
                                           0.2)
        else:
            self.hover_color[-1] = 0.2
            self.stationary_pos = hand_pos
            self.stationary_start_time = 0.0
