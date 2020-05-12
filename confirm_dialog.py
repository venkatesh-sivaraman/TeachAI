from kivy.properties import *
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.uix.label import Label
from kivy.clock import Clock
from paintbrush import *
import numpy as np
from functools import partial
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image
import io
from PIL import Image as PILImage
import colorsys

PAINTBRUSH_COLORS = [colorsys.hsv_to_rgb(*item) for item in [
    (0, 0.8, 0.9),
    (0.3, 0.8, 0.9),
    (0.6, 0.8, 0.9),
    (0.9, 0.8, 0.9),
    (0.2, 0.8, 0.9),
    (0.5, 0.8, 0.9),
    (0.8, 0.8, 0.9),
    (0.1, 0.8, 0.9),
    (0.4, 0.8, 0.9),
    (0.7, 0.8, 0.9)
]]

class ConfirmDialog(Widget):
    image_src = StringProperty("")

    # TODO show a better representation of the image
    transcript = StringProperty("")

    labels = ListProperty([])

    def __init__(self, **kwargs):
        super(ConfirmDialog, self).__init__(**kwargs)
        self.current_time = 0.0
        self.on_confirm = None
        self.paintbrushes = []
        self.label_widgets = []
        Clock.schedule_once(self.initialize, 0.1)

    def initialize(self, dt):
        self.ids.done_button.set_hover_rgb(0, 0.8, 0.9)
        self.ids.redo_button.set_hover_rgb(1, 0, 0)
        self.ids.done_button.hold_time = 3.0
        self.ids.redo_button.hold_time = 3.0
        self.ids.done_button.on_click = self.on_done_button_pressed
        self.ids.redo_button.on_click = self.on_redo_button_pressed

    def update(self, dt):
        self.current_time += dt
        self.ids.done_button.update(dt)
        self.ids.redo_button.update(dt)

    def on_labels(self, instance, value):
        self.reset()
        image_view = self.ids.image_view

        for i, label in enumerate(value):
            # Make an image for this mask
            image = Image(source="", keep_ratio=True, allow_stretch=True)
            imgIO = io.BytesIO()
            mask_img = np.zeros((*label.mask.shape, 4))
            mask_img[:,:,:3] = np.array(PAINTBRUSH_COLORS[i % len(PAINTBRUSH_COLORS)])
            mask_img[:,:,3] = np.where(np.isnan(label.mask), 0.0, label.mask) * 0.6
            pil_img = PILImage.fromarray((mask_img * 255).astype(np.uint8),
                                         mode='RGBA').resize(image_view.image_size)
            pil_img.save(imgIO, format='png')
            imgIO.seek(0)
            imgData = io.BytesIO(imgIO.read())
            image.texture = CoreImage(imgData, ext='png').texture
            image.reload()

            self.ids.paintbrush_parent.add_widget(image)
            self.paintbrushes.append(image)
            image.pos = self.ids.paintbrush_parent.pos
            image.size = self.ids.paintbrush_parent.size

            for center, stroke in zip(label.centers, label.strokes):
                # Label
                center = image_view.convert_point_from_image(center)
                lw = Label(pos=(int(center[0] + self.ids.label_parent.pos[0]),
                                int(center[1] + self.ids.label_parent.pos[1])),
                           text=label.label)
                self.label_widgets.append(lw)
                self.ids.label_parent.add_widget(lw)

    def reset(self):
        for paintbrush in self.paintbrushes:
            self.ids.paintbrush_parent.remove_widget(paintbrush)
        self.paintbrushes = []
        for lw in self.label_widgets:
            self.ids.label_parent.remove_widget(lw)
        self.label_widgets = []

    def update_palm(self, hand_pos, depth):
        if hand_pos is None or self.disabled:
            return
        self.ids.redo_button.update_palm(hand_pos, depth)
        self.ids.done_button.update_palm(hand_pos, depth)

    def on_done_button_pressed(self):
        self.on_confirm(True)

    def on_redo_button_pressed(self):
        self.on_confirm(False)
