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
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import *
from kivy.clock import Clock
from image_controller import ImageController
from speech import SpeechWidget
from confirm_dialog import ConfirmDialog
from button import LeapButton
from point_util import *
from classification import *
import numpy as np
import datetime
import pickle

# controller = Leap.Controller()
IMAGE_BASE = "/Users/venkatesh-sivaraman/Documents/School/MIT/6-835/coco/val2017"

class MainWidget(FloatLayout):
    cursor_pos = ListProperty([0, 0])
    cursor_size = NumericProperty(0.0)

    image_sources = []
    image_idx = 0

    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        with open("coco_cat_dog_imgs.txt", "r") as file:
            fnames = [line.strip() for line in file.readlines()]
        self.image_sources = [os.path.join(IMAGE_BASE, fn) for fn in fnames]
        random.shuffle(self.image_sources)
        self.ids.image_controller.image_src = self.image_sources[0]
        self.image_idx += 1
        self.ids.image_controller.bind(segmentation=self.image_controller_completed)
        self.current_data = None
        self.model = ClassificationModel("/Users/venkatesh-sivaraman/Downloads/small-bigger-autoencoder-09-0.0029.hdf5")
        # self.ids.confirm_dialog.on_confirm = self.confirm_dialog_completed

    def update(self, dt):
        self.ids.image_controller.update(dt)
        self.ids.confirm_dialog.update(dt)

    def update_palm(self, hand_pos):
        if hand_pos is None:
            self.ids.image_controller.update_palm(None, 0.0)
            self.ids.confirm_dialog.update_palm(None, 0.0)
            return

        # Compute cursor position and size
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
        self.ids.image_controller.update_palm(self.cursor_pos,
                                              self.cursor_size / cursor_max)
        self.ids.confirm_dialog.update_palm(self.cursor_pos,
                                              self.cursor_size / cursor_max)

    def image_controller_completed(self, instance, value):
        if not value:
            return
#         self.current_data = value
#         self.ids.confirm_dialog.opacity = 1.0
#         self.ids.confirm_dialog.disabled = False
#         self.ids.confirm_dialog.image_src = self.ids.image_controller.image_src
#         self.ids.confirm_dialog.transcript = "\"{}\"".format(value['speech']['transcript'])
# 
#     def confirm_dialog_completed(self, confirmed):
#         if confirmed:
        out_path = os.path.join("fusion_data", "{}_{}.pkl".format(
            os.path.basename(self.ids.image_controller.image_src),
            datetime.datetime.now()
        ))
        with open(out_path, "wb") as file:
            pickle.dump(value, file)
        self.model.add_training_example(self.ids.image_controller.image_src,
                                        [ann['gesture'] for ann in value])

        print("Completed")
        self.ids.image_controller.image_src = self.image_sources[self.image_idx]
        if self.image_idx >= 5:
            self.model.predict_training_example(self.image_sources[self.image_idx])
        self.image_idx += 1

        # self.ids.confirm_dialog.opacity = 0.0
        # self.ids.confirm_dialog.disabled = True
        self.ids.image_controller.reset()

class MainApp(App):

    def on_start(self):
        self.event = Clock.schedule_interval(self.update, 1 / 30.)

    def on_stop(self):
        Clock.unschedule(self.event)

    def build(self):
        self.main_widget = MainWidget()
        return self.main_widget

    def update(self, dt):
        self.main_widget.update(dt)
        frame = getLeapFrame()
        # frame = controller.frame()
        loc = frame.hands[0].palm_pos # leap_one_palm(frame)
        if not all(x == 0 for x in loc):
            self.main_widget.update_palm(loc)
        else:
            self.main_widget.update_palm(None)


if __name__ == '__main__':
    MainApp().run()
