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
    left_color = ListProperty([1, 0, 0, 0.0])
    right_color = ListProperty([0, 0.8, 0.9, 0.0])
    hold_time = 3.0
    image_src = StringProperty("")

    # TODO show a better representation of the image
    transcript = StringProperty("")

    labels = ListProperty([])

    def __init__(self, **kwargs):
        super(ConfirmDialog, self).__init__(**kwargs)
        self.stationary_start_time = 0.0
        self.stationary_pos = None
        self.current_time = 0.0
        self.on_confirm = None
        self.paintbrushes = []
        self.label_widgets = []

    def update(self, dt):
        self.current_time += dt

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
                # Add a paintbrush
                # paintbrush = Paintbrush()
                # paintbrush.rgb = PAINTBRUSH_COLORS[i % len(PAINTBRUSH_COLORS)]
                # paintbrush.pos_hint = {'x': 0, 'y': 0}
                # paintbrush.size_hint = (1, 1)
                # self.ids.paintbrush_parent.add_widget(paintbrush)
                # self.paintbrushes.append(paintbrush)

                # def paint(paintbrush, dt):
                #     alpha = np.hanning(len(stroke))
                #     for k, point in enumerate(stroke):
                #         image_point = image_view.convert_point_from_image((point[0], point[1]))
                #         width = image_view.convert_dimension_from_image(point[2])
                #         paintbrush.add_point((image_point[0] + image_view.pos[0],
                #                               image_point[1] + image_view.pos[1]), width=width,
                #                              alpha=alpha[k])
                # Clock.schedule_once(partial(paint, paintbrush), 0.1)
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
