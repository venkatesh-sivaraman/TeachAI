from leaputil import *
import Leap

import time
import kivy
from kivy.app import App
from kivy.clock import Clock
from image_controller import ImageController

controller = Leap.Controller()

class MainApp(App):

    def on_start(self):
        self.event = Clock.schedule_interval(self.update, 1 / 30.)

    def on_stop(self):
        Clock.unschedule(self.event)

    def build(self):
        self.image_controller = ImageController()
        return self.image_controller

    def update(self, dt):
        frame = controller.frame()
        loc = leap_one_palm(frame)
        if not all(x == 0 for x in loc):
            self.image_controller.update_palm(loc)


if __name__ == '__main__':
    MainApp().run()
