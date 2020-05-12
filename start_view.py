from kivy.properties import *
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.uix.label import Label
from kivy.clock import Clock
import numpy as np
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image
import io
from PIL import Image as PILImage

class StartView(Widget):
    def __init__(self, **kwargs):
        super(StartView, self).__init__(**kwargs)
        self.on_start = None
        self.current_time = 0.0
        Clock.schedule_once(self.initialize, 0.1)

    def initialize(self, dt):
        self.ids.start_button.set_hover_rgb(0, 0.8, 0)
        self.ids.start_button.hold_time = 2.0
        self.ids.start_button.on_click = self.on_start_button_pressed

    def update(self, dt):
        if self.disabled: return
        self.current_time += dt
        self.ids.start_button.update(dt)

    def on_start_button_pressed(self):
        if self.on_start:
            self.on_start()

    def update_palm(self, hand_pos, depth):
        if self.disabled: return
        if hand_pos is None:
            return

        self.ids.start_button.update_palm(hand_pos, depth)
