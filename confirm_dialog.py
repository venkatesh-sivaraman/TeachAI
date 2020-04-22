from kivy.properties import *
from kivy.uix.widget import Widget
import numpy as np

class ConfirmDialog(Widget):
    left_color = ListProperty([1, 0, 0, 0.0])
    right_color = ListProperty([0, 0.8, 0.9, 0.0])
    hold_time = 3.0
    image_src = StringProperty("")

    # TODO show a better representation of the image
    transcript = StringProperty("")

    def __init__(self, **kwargs):
        super(ConfirmDialog, self).__init__(**kwargs)
        self.stationary_start_time = 0.0
        self.stationary_pos = None
        self.current_time = 0.0
        self.on_confirm = None

    def update(self, dt):
        self.current_time += dt

    def update_palm(self, hand_pos, depth):
        if hand_pos is None or self.disabled:
            return
        hand_right = hand_pos[0] >= self.size[0] / 2
        hand_pos = np.array(hand_pos)
        if self.stationary_pos is not None and np.linalg.norm(hand_pos - self.stationary_pos) <= 60.0:
            if self.stationary_start_time == 0.0:
                self.stationary_start_time = self.current_time
            elif self.current_time - self.stationary_start_time >= self.hold_time:
                if self.on_confirm:
                    self.on_confirm(hand_right)
            else:
                time = self.current_time - self.stationary_start_time
                if hand_right:
                    self.right_color[-1] = max((time - self.hold_time / 2) / (self.hold_time / 2), 0)
                else:
                    self.left_color[-1] = max((time - self.hold_time / 2) / (self.hold_time / 2), 0)
        else:
            self.right_color[-1] = 0.0
            self.left_color[-1] = 0.0
            self.stationary_pos = hand_pos
            self.stationary_start_time = 0.0
