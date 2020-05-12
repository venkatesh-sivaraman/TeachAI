# from leaputil import *
# import Leap
import sys
sys.path.insert(0, "pyleap")
from pyleap.leap import getLeapInfo, getLeapFrame
import os
import random

import matplotlib.pyplot as plt
import PIL

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
from paintbrush import *
from classification import *
from start_view import *
import numpy as np
import datetime
import pickle

PRETRAINING_DIR = os.path.abspath("pretraining")
IMAGE_BASE = os.path.abspath("training_data")
OUTPUT_DIR = "fusion_output"

class MainWidget(FloatLayout):
    cursor_pos = ListProperty([0, 0])
    cursor_size = NumericProperty(0.0)

    image_sources = []
    image_idx = 0

    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        self.ids.image_controller.bind(segmentation=self.image_controller_completed)
        self.current_data = None
        self.model = ClassificationModel(PRETRAINING_DIR, IMAGE_BASE)
        self.ids.confirm_dialog.on_confirm = self.confirm_dialog_completed

        # Present the first image
        with open(os.path.join(IMAGE_BASE, "filenames.txt"), "r") as file:
            fnames = [line.strip() for line in file.readlines()]
        self.image_sources = [os.path.join(IMAGE_BASE, fn) for fn in fnames]
        random.shuffle(self.image_sources)

        self.start_screen_finished = False
        self.ids.start_view.on_start = self.dismiss_start_view

    def dismiss_start_view(self):
        self.start_screen_finished = True
        self.ids.start_view.opacity = 0.0
        self.ids.start_view.disabled = True
        while self.model.has_used_image(self.image_sources[self.image_idx]):
            self.image_idx += 1
        self.present_image(self.image_sources[self.image_idx])
        self.image_idx += 1

    def update(self, dt):
        if self.start_screen_finished:
            self.ids.image_controller.update(dt)
        self.ids.confirm_dialog.update(dt)
        self.ids.start_view.update(dt)

    def present_image(self, image_source):
        print(image_source)
        self.ids.image_controller.image_src = image_source
        new_prompt = "How do I tell what dog breed this is?"

        predicted_label, confidence, labels, masks, total_mask = self.model.predict_image(image_source)
        mask = None

        if confidence >= 0.5:
            new_prompt = ("I think this is a {} because of {} and {} shown here. "
                          "Can you explain if that's right?").format(
                predicted_label, labels[0][0], labels[1][0])
            mask = (masks[labels[0][0]] + masks[labels[1][0]]) ** 2
        elif confidence >= 0.3:
            new_prompt = ("This might be a {} because of these regions but I'm not sure. "
                          "Can you explain the answer?".format(predicted_label))
            mask = total_mask
        if mask is not None:
            mask_colored = np.zeros((*mask.shape, 4))
            mask_colored[:,:,:3] = np.array([0, 0, 1])
            mask_colored[:,:,3] = mask / np.max(mask)
            original_image = PIL.Image.open(image_source)
            overlay = PIL.Image.fromarray((mask_colored * 255.).astype(np.uint8)).resize(original_image.size)
            overlay.save("tmp/overlay.png")
            overlay = os.path.abspath("tmp/overlay.png")
        else:
            overlay = None

        def cb(dt):
            self.ids.image_controller.update_prompt(new_prompt, overlay)
        Clock.schedule_once(cb, 0.3)

    def update_palm(self, hand_pos):
        if hand_pos is None:
            self.ids.image_controller.update_palm(None, 0.0)
            self.ids.confirm_dialog.update_palm(None, 0.0)
            self.ids.start_view.update_palm(None, 0.0)
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
        self.cursor_size = float(cursor_max - np.clip(new_size, 0, cursor_max))
        if self.start_screen_finished:
            self.ids.image_controller.update_palm(self.cursor_pos,
                                                  self.cursor_size / cursor_max)
        self.ids.confirm_dialog.update_palm(self.cursor_pos,
                                              self.cursor_size / cursor_max)
        self.ids.start_view.update_palm(self.cursor_pos,
                                        self.cursor_size / cursor_max)

    def on_palm_close_gesture(self):
        self.ids.image_controller.on_palm_close_gesture()

    def image_controller_completed(self, instance, value):
        if not value:
            return
        self.current_data = value
        self.ids.confirm_dialog.opacity = 1.0
        self.ids.confirm_dialog.disabled = False
        self.ids.confirm_dialog.image_src = self.ids.image_controller.image_src
        self.ids.confirm_dialog.transcript = "\"{}\"".format(value['speech']['transcript'])
        self.ids.confirm_dialog.labels = value['labels']

    def confirm_dialog_completed(self, confirmed):
        if confirmed:
            self.model.add_training_point(self.ids.image_controller.image_src, self.current_data['labels'])
            if not os.path.exists(OUTPUT_DIR):
                os.mkdir(OUTPUT_DIR)
            out_path = os.path.join(OUTPUT_DIR, "{}_{}.pkl".format(
                os.path.basename(self.ids.image_controller.image_src),
                datetime.datetime.now()
            ))
            with open(out_path, "wb") as file:
                pickle.dump(self.current_data, file)

            print("Completed")
            while self.model.has_used_image(self.image_sources[self.image_idx]):
                self.image_idx += 1
            self.present_image(self.image_sources[self.image_idx])
            self.image_idx += 1

        self.ids.confirm_dialog.opacity = 0.0
        self.ids.confirm_dialog.disabled = True
        self.ids.confirm_dialog.reset()
        self.ids.image_controller.reset()

class MainApp(App):

    def on_start(self):
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        self.event = Clock.schedule_interval(self.update, 1 / 30.)
        self.finger_history = []
        self.time_since_palm_close = -1

    def on_stop(self):
        Clock.unschedule(self.event)

    def build(self):
        self.main_widget = MainWidget()
        return self.main_widget

    def update(self, dt):
        self.main_widget.update(dt)
        frame = getLeapFrame()
        loc = frame.hands[0].palm_pos

        if not all(x == 0 for x in loc):
            self.main_widget.update_palm(loc)
        else:
            self.main_widget.update_palm(None)


if __name__ == '__main__':
    MainApp().run()
