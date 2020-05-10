import kivy
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import *
from kivy.graphics.instructions import InstructionGroup
from kivy.clock import Clock
from kivy.graphics import Ellipse, Color, Rectangle
import numpy as np
from point_util import *
from region_detector import *
from classification import *
from labeling import *
from paintbrush import *
import PIL
import time

class ImageContainer(FloatLayout):
    image_source = StringProperty('test_image.jpg')
    overlay_source = StringProperty('', allownone=True)
    image_size = ListProperty([0, 0])

    def on_image_source(self, instance, value):
        if value:
            img = PIL.Image.open(value)
            self.image_size = img.size
        else:
            self.image_size = [0, 0]

    def convert_point_to_image(self, point):
        aspect_ratio = max(self.image_size[0] / self.size[0], self.image_size[1] / self.size[1])
        start_x = self.size[0] / 2.0 - self.image_size[0] / 2.0 / aspect_ratio
        start_y = self.size[1] / 2.0 - self.image_size[1] / 2.0 / aspect_ratio
        result = ((point[0] - start_x) * aspect_ratio,
                self.image_size[1] - (point[1] - start_y) * aspect_ratio)
        return result

    def convert_point_from_image(self, point):
        aspect_ratio = max(self.image_size[0] / self.size[0], self.image_size[1] / self.size[1])
        start_x = self.size[0] / 2.0 - self.image_size[0] / 2.0 / aspect_ratio
        start_y = self.size[1] / 2.0 - self.image_size[1] / 2.0 / aspect_ratio
        return (point[0] / aspect_ratio + start_x,
                (self.image_size[1] - point[1]) / aspect_ratio + start_y)

    def convert_dimension_to_image(self, dim):
        """Converts a single value to image coordinates."""
        aspect_ratio = max(self.image_size[0] / self.size[0], self.image_size[1] / self.size[1])
        return dim * aspect_ratio

    def convert_dimension_from_image(self, dim):
        """Converts a single value from image coordinates."""
        aspect_ratio = max(self.image_size[0] / self.size[0], self.image_size[1] / self.size[1])
        return dim / aspect_ratio

label_colors = [
    (1, 0, 0, 0.3),
    (0, 1, 0, 0.3),
    (0, 0.6, 1, 0.3)
]
temporary_color = (0.9, 0.9, 0.9, 0.8)
known_labels = []

class AnnotationsView(Widget):
    annotations = ListProperty([])
    image_size = ListProperty([0, 0])

    def __init__(self, **kwargs):
        super(AnnotationsView, self).__init__(**kwargs)
        self.labels = {}

    def on_annotations(self, instance, value):
        self._update_annotations(value)

    def annotations_updated(self):
        self._update_annotations(self.annotations)

    def _update_annotations(self, value):
        self.canvas.clear()

        # Compute origin of image based on aspect ratio
        aspect_ratio = max(self.image_size[0] / self.size[0], self.image_size[1] / self.size[1])
        start_x = self.pos[0] + self.size[0] / 2.0 - self.image_size[0] / 2.0 / aspect_ratio
        start_y = self.pos[1] + self.size[1] / 2.0 + self.image_size[1] / 2.0 / aspect_ratio
        with self.canvas:
            for region in self.annotations:
                tile_width = self.image_size[0] / aspect_ratio / region.mask.shape[1]
                tile_height = self.image_size[1] / aspect_ratio / region.mask.shape[0]
                if region.temporary:
                    Color(*temporary_color)
                else:
                    if region.label:
                        if region.label not in known_labels:
                            known_labels.append(region.label)
                        # Add a text label
                        if region not in self.labels:
                            nonzero_locs = np.argwhere(region.mask)
                            center = np.mean(nonzero_locs, axis=0)
                            label = Label(pos=(int(start_x + center[1] * tile_width),
                                               int(start_y - (center[0] + 1) * tile_height)),
                                          text=region.label)
                            self.add_widget(label)
                            self.labels[region] = label
                    Color(*label_colors[known_labels.index(region.label) % len(label_colors)])

                for y in range(region.mask.shape[0]):
                    for x in range(region.mask.shape[1]):
                        if region.mask[y,x] == 0: continue
                        Rectangle(pos=(start_x + x * tile_width,
                                       start_y - (y + 1) * tile_height),
                                  size=(tile_width, tile_height))

        # Remove old labels
        for r, l in self.labels.items():
            if r not in value:
                self.remove_widget(l)
        self.labels = {r: l for r, l in self.labels.items() if r in value}

class GestureDrawing(Widget):

    points = ListProperty([])

    def __init__(self, **kwargs):
        super(GestureDrawing, self).__init__(**kwargs)

    def add_point(self, point):
        for elem in point:
            self.points.append(elem)

CURSOR_TRAIL_LENGTH = 20
CURSOR_RGB = (0.98, 0.85, 0.16)
CURSOR_ON_RGB = (0.23, 0.9, 0.99)
CURSOR_ELLIPSE_SIZE = 30.0

class CursorTrail(InstructionGroup):
    points = []
    colors = []
    on_colors = []

    def __init__(self, **kwargs):
        super(CursorTrail, self).__init__(**kwargs)
        # Create a trail of colors
        for opacity in np.linspace(0, 1, CURSOR_TRAIL_LENGTH):
            self.colors.append(Color(*CURSOR_RGB, opacity))
            self.on_colors.append(Color(*CURSOR_ON_RGB, opacity))

    def update_instructions(self):
        self.clear()
        for i, (point, active) in enumerate(self.points):
            self.add(self.colors[i] if not active else self.on_colors[i])
            self.add(point)

    def add_point(self, point, radius, is_active):
        self.points.append((Ellipse(pos=(point[0] - radius,
                                        point[1] - radius),
                                   size=(radius * 2, radius * 2)), is_active))
        if len(self.points) > CURSOR_TRAIL_LENGTH:
            self.points.pop(0)
        self.update_instructions()

    def pop_point(self):
        if not self.points:
            return
        self.points.pop(0)
        self.update_instructions()

HOVER_TOLERANCE = 20.0

class ImageController(FloatLayout):

    segmentation = ObjectProperty(None, allownone=True)
    image_src = StringProperty("test_image.jpg")
    associations = ListProperty([])

    prompt = StringProperty("")

    def __init__(self, **kwargs):
        super(ImageController, self).__init__(**kwargs)
        self.gesture_points = []
        #self.cursor_trail = None
        self.current_annotation = None
        self.current_time = 0.0
        self.finished_region = None
        self.previous_points = []
        Clock.schedule_once(self.initialize, 0.1)

    def initialize(self, dt):
        self.ids.speech_widget.bind(transcript=self.speech_transcript_completed)
        self.ids.speech_widget.on_silence = self.finish_association
        self.ids.done_button.set_hover_rgb(0, 0.8, 0.9)
        self.ids.done_button.on_click = self.on_done_button_pressed
        self.ids.redo_button.set_hover_rgb(1, 0, 0)
        self.ids.redo_button.on_click = self.on_redo_button_pressed
        self.ids.image_view.bind(image_size=self.on_image_size_changed)
        self.ids.annotations_view.image_size = self.ids.image_view.image_size
        with self.canvas:
            #BindTexture(source="test_image.jpg")
            Color(0, 1, 1, 0.7)
            self.cursor_halo = Rectangle(pos=(0, 0), size=(100, 100), source='blur_circle.png')

    def on_image_size_changed(self, instance, value):
        self.ids.annotations_view.image_size = value

    def reset(self):
        self.segmentation = None
        self.current_time = 0.0
        self.ids.speech_widget.reset()
        self.previous_points = []
        self.ids.image_view.overlay_source = ""
        self.ids.paintbrush.clear()

    def on_image_src(self, instance, value):
        if not value:
            return
        self.ids.annotations_view.annotations = []
        self.current_annotation = None
        self.gesture_points = []
        self.associations = []

    def on_prompt(self, instance, value):
        if not value:
            return
        def cb(dt):
            recording = self.ids.speech_widget.is_recording
            if recording:
                self.ids.speech_widget.pause_recording()
            self.speak(value)
            if recording:
                self.ids.speech_widget.resume_recording()
        Clock.schedule_once(cb, 0.1)

    def update(self, dt):
        self.ids.done_button.update(dt)
        self.ids.redo_button.update(dt)
        self.ids.speech_widget.update(dt)
        self.current_time += dt
        if not self.ids.speech_widget.is_recording:
            self.current_time = 0.0
            self.ids.speech_widget.start_recording()
        self.ids.done_button.disabled = (not self.ids.speech_widget.had_sound)
        if self.ids.speech_widget.is_transcribing:
            self.ids.done_button.text = "Transcribing..."
        else:
            self.ids.done_button.text = "I'm done with this image"

    def update_palm(self, hand_pos, depth):
        self.ids.done_button.update_palm(hand_pos, depth)
        self.ids.redo_button.update_palm(hand_pos, depth)
        if hand_pos is None:
            #if self.cursor_trail:
            #    self.cursor_trail.pop_point()
            self.previous_points = []
            if self.current_annotation:
                self.current_annotation = None
                del self.ids.annotations_view.annotations[-1]
            self.gesture_points = []
            return

        image_pos = (hand_pos[0] - self.ids.image_view.pos[0],
                     hand_pos[1] - self.ids.image_view.pos[1]) #convert_point(hand_pos, self, self.ids.image_view)

        # if len(self.gesture_points) % 10 == 0 and len(self.gesture_points) > 10:
        #     region = detect_region([x[0] for x in self.gesture_points],
        #                            self.ids.image_view.image_size,
        #                           (14, 14))
        #     region = LabeledRegion(region, "", temporary=True)
        #     if self.current_annotation is None:
        #         self.current_annotation = region
        #         self.ids.annotations_view.annotations.append(self.current_annotation)
        #     else:
        #         self.current_annotation = region
        #         self.ids.annotations_view.annotations[-1] = self.current_annotation

        # if not self.cursor_trail:
        #     self.cursor_trail = CursorTrail()
        #     self.ids.image_view.canvas.add(self.cursor_trail)
        # self.cursor_trail.add_point(hand_pos, 20.0, len(self.gesture_points) > 0)
        if self.segmentation or self.ids.speech_widget.is_transcribing: # finished
            self.cursor_halo.size = (0, 0)
        else:
            line_width = self.ids.paintbrush.add_point(hand_pos, 1 - depth)
            image_coord_line_width = self.ids.image_view.convert_dimension_to_image(line_width)
            self.gesture_points.append(((*self.ids.image_view.convert_point_to_image(image_pos),
                                         1 - depth, image_coord_line_width), self.current_time))

            # Add cursor halo
            self.cursor_halo.pos = (hand_pos[0] - line_width / 2,
                                    hand_pos[1] - line_width / 2)
            self.cursor_halo.size = (line_width, line_width)

        # Check for hold gesture
        # self.previous_points.append((self.current_time, hand_pos))
        # if self.previous_points[-1][0] - self.previous_points[0][0] > 1.5:
        #     self.previous_points.pop(0)
        # if (self.previous_points[-1][0] - self.previous_points[0][0] >= 1.0 and
        #     self.current_time >= 3.0 and
        #     self.ids.speech_widget.had_sound and
        #     len(self.gesture_points) > 10): # Make sure it doesn't trigger stopping immediately
        #     xs = np.array([item[1][0] for item in self.previous_points])
        #     ys = np.array([item[1][1] for item in self.previous_points])
        #     if (np.max(xs) - np.min(xs) <= HOVER_TOLERANCE and
        #         np.max(ys) - np.min(ys) <= HOVER_TOLERANCE):
        #         self.finish_association()

    def on_palm_close_gesture(self):
        if not self.ids.speech_widget.had_sound:
            self.speak("Can you describe this image?")
            return
        self.finish_association()

    def speak(self, message):
        self.ids.speech_widget.pause_recording()
        os.system('say "{}"'.format(message))
        self.ids.speech_widget.resume_recording()

    def finish_association(self):
        if len(self.gesture_points) > 20:
            region = detect_region([x[0] for x in self.gesture_points],
                                   self.ids.image_view.image_size,
                                  (14, 14))
            annotation  = LabeledRegion(region, "", temporary=True)
            old_gesture_points = self.gesture_points
            start_time = 0
            def cb(value):
                self.ids.transcribing_overlay.opacity = 0.0
                self.ids.transcribing_overlay.disabled = True

                if not value or not value['transcript']:
                    return
                # region = detect_region([x[0] for x in self.old_gesture_points],
                #                        self.ids.image_view.image_size,
                #                       (14, 14))
                timestamped_points = [(x[1] - start_time, x[0][0], x[0][1], x[0][3]) for x in old_gesture_points]
                labels = label_region(timestamped_points, value, self.ids.image_view.image_size)
                self.segmentation = {
                    'labels': labels,
                    'speech': value
                }
                self.ids.transcribing_overlay.opacity = 0.0
                self.ids.transcribing_overlay.disabled = True
                # label = self.get_label(value['transcript'])
                # if not label:
                #     self.ids.annotations_view.annotations.remove(annotation)
                #     return
                # annotation.label = label
                # annotation.temporary = False
                # self.ids.annotations_view.annotations_updated()
                # self.associations.append({'speech': value, 'gesture': annotation,
                #                           'gesture_points': old_gesture_points})

            start_time = self.ids.speech_widget.stop_recording(True, cb)
        else:
            self.ids.speech_widget.stop_recording(False)
        self.gesture_points = []
        self.current_annotation = None
        self.ids.transcribing_overlay.opacity = 1.0
        self.ids.transcribing_overlay.disabled = False
        # self.reset()

    def get_label(self, transcript):
        nouns = list(find_nouns(transcript))
        if len(nouns) > 1:
            os.system('say "I heard {}. Can you use just one noun this time?"'.format(' or '.join(nouns)))
            return None
        elif len(nouns) == 0:
            os.system("say \"I didn't catch what that was. Can you try again?\"")
            return None

        label = nouns[0]
        if label not in known_labels:
            message_idx = np.random.choice(3)
            if message_idx == 0:
                os.system("say \"I've never seen a {} before. Cool!\"".format(label))
            elif message_idx == 1:
                os.system("say \"that's a {}. good to know!\"".format(label))
            else:
                os.system("say \"so that's what a {} looks like.\"".format(label))
        return label

    def on_done_button_pressed(self):
        self.finish_association()
        # self.segmentation = list(self.associations)

    def on_redo_button_pressed(self):
        self.ids.annotations_view.annotations = []
        self.current_annotation = None
        self.gesture_points = []
        self.associations = []
        self.reset()

    def speech_transcript_completed(self, instance, value):
        self.ids.transcribing_overlay.opacity = 0.0
        self.ids.transcribing_overlay.disabled = True

        if not value or not value['transcript']:
            return
        # region = detect_region([x[0] for x in self.old_gesture_points],
        #                        self.ids.image_view.image_size,
        #                       (14, 14))
        timestamped_points = [(x[1], x[0][0], x[0][1], x[0][3]) for x in self.old_gesture_points]
        label_region(timestamped_points, value, self.ids.image_view.image_size)

        # Determine what the most likely label is
        # nouns = list(find_nouns(value['transcript']))
        # if len(nouns) > 1:
        #     os.system('say "I heard {}. Can you use just one noun this time?"'.format(' or '.join(nouns)))
        #     self.reset()
        #     if self.old_annotation:
        #         self.ids.annotations_view.annotations.remove(self.old_annotation)
        #         self.old_annotation = None
        #     return
        # elif len(nouns) == 0:
        #     os.system("say \"I didn't catch what that was. Can you try again?\"")
        #     self.reset()
        #     if self.old_annotation:
        #         self.ids.annotations_view.annotations.remove(self.old_annotation)
        #         self.old_annotation = None
        #     return
        # label = nouns[0]
        # if label not in known_labels:
        #     message_idx = np.random.choice(3)
        #     if message_idx == 0:
        #         os.system("say \"I've never seen a {} before. Cool!\"".format(label))
        #     elif message_idx == 1:
        #         os.system("say \"that's a {}. good to know!\"".format(label))
        #     else:
        #         os.system("say \"so that's what a {} looks like.\"".format(label))

        # region = LabeledRegion(region, label)
        # if self.old_annotation:
        #     self.ids.annotations_view.annotations.remove(self.old_annotation)
        #     self.old_annotation = None
        # if self.current_annotation:
        #     del self.ids.annotations_view.annotations[-1]
        # self.ids.annotations_view.annotations.append(region)
        # if self.current_annotation:
        #     self.ids.annotations_view.annotations.append(self.current_annotation)
        # self.associations.append({'speech': value, 'gesture': region,
        #                           'gesture_points': self.gesture_points})
        # self.gesture_points = []
        # self.reset()


