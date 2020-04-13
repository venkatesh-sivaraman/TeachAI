# from leaputil import *
# import Leap
import sys
sys.path.insert(0, "pyleap")
from pyleap.leap import getLeapInfo, getLeapFrame
import os
import random

import time
import kivy
from kivy.app import App
from kivy.clock import Clock
from image_controller import ImageController
from speech import SpeechWidget

# controller = Leap.Controller()
IMAGE_BASE = "/Users/venkatesh-sivaraman/Documents/School/MIT/6-835/coco/val2017"

class MainApp(App):

    image_sources = []
    image_idx = 0

    def on_start(self):
        self.event = Clock.schedule_interval(self.update, 1 / 30.)

    def on_stop(self):
        Clock.unschedule(self.event)

    def build(self):
        self.image_sources = [os.path.join(IMAGE_BASE, fn) for fn in os.listdir(IMAGE_BASE)
                              if not fn.startswith(".") and fn.endswith("jpg")]
        random.shuffle(self.image_sources)
        self.image_controller = ImageController()
        self.image_controller.image_src = self.image_sources[0]
        self.image_idx += 1
        self.image_controller.bind(segmentation=self.image_controller_completed)
        return self.image_controller

    def update(self, dt):
        self.image_controller.update(dt)
        frame = getLeapFrame()
        # frame = controller.frame()
        loc = frame.hands[0].palm_pos # leap_one_palm(frame)
        if not all(x == 0 for x in loc):
            self.image_controller.update_palm(loc)
        else:
            self.image_controller.update_palm(None)

    def image_controller_completed(self, instance, value):
        if not value:
            return
        print("Completed")
        time.sleep(1)
        self.image_controller.image_src = self.image_sources[self.image_idx]
        self.image_idx += 1

if __name__ == '__main__':
    MainApp().run()
