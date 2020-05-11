from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

exclude_words = [
    "top", # TODO can we improve region detection using these words?
    "bottom",
    "left",
    "right",
    "middle",
    "frame",
    "picture",
    "photo",
    "region",
    "part",
    "section",
    "area"
    "correct",
    "wrong",
    "incorrect",
    "well",
    "good",
    "bad",
    "guess",
    "try",
    "explanation",
    "reason",
    "yes",
    "no",
    "other",
    "another"
]

class LabeledMask:
    def __init__(self, label, mask, centers, strokes):
        """
        label: A string describing what is in the given mask of the image.
        mask: A small k x k image that contains mask values from 0 to 1.
        centers: A list of points in the original image coordinates representing the centers of the
            label
        strokes: A list of lists of (x, y, width) locations corresponding to this label.
        """
        self.label = label
        self.mask = mask
        self.centers = centers
        self.strokes = strokes

lemmatizer = WordNetLemmatizer()

WORD_TIMESTAMP_MARGIN = 1.0 # seconds around the word start and end to grab as part of the label
DEBUG = False

def find_nouns(phrase):
    tokens = nltk.word_tokenize(phrase)
    pos = nltk.pos_tag(tokens)
    return set([lemmatizer.lemmatize(word) for word, p in pos if p.startswith("NN") and
                word.lower() not in exclude_words])

def extend_word_timestamps(timestamps):
    """Returns a new list of word timestamps, where the ends of each word have been extended to
    the beginning of the next word."""
    result = []
    for i, token in enumerate(timestamps):
        result.append({
            "word": token["word"],
            "start_time": token["start_time"],
            "end_time": (timestamps[i + 1]["start_time"] if i < len(timestamps) - 1 else
                         token["end_time"])
        })
    return result

def word_tokenize_timestamps(timestamps):
    """Returns a new set of timestamps where any compound words are tokenized as needed."""
    result = []
    for token in timestamps:
        nl_tokens = nltk.word_tokenize(token["word"])
        for nlt in nl_tokens:
            result.append({
                "word": nlt,
                "start_time": token["start_time"],
                "end_time": token["end_time"]
            })
    return result

def label_region(gesture_points, transcript, image_size):
    """
    gesture_points: a list of (t, x, y, width) points describing the path of the user's hand relative to
        the image coordinate system in time (seconds relative to the start of recording).
    transcript: an object representing the result of a google speech API call.
    """

    t = [x[0] for x in gesture_points]
    y = np.array([(x[1], x[2], x[3]) for x in gesture_points]).T
    trajectory = interp1d(t, y, kind='linear', fill_value="extrapolate")

    ges_x = [x[1] for x in gesture_points]
    ges_y = [x[2] for x in gesture_points]
    ges_widths = np.array([x[3] for x in gesture_points])

    timestamps = word_tokenize_timestamps(transcript["timestamps"])
    pos_tags = nltk.pos_tag([token["word"] for token in timestamps])

    # Sometimes the timestamps get out of sync
    if abs(timestamps[-1]["end_time"] - t[-1]) >= 5.0:
        return []

    if DEBUG:
        print(timestamps, t)
        plt.figure()
        plt.scatter(ges_x, ges_y, color='r', s=ges_widths)
        plt.plot(ges_x, ges_y, linestyle='-', color='r')
        for token, (_, pos) in zip(timestamps, pos_tags):
            sub_x, sub_y, _ = trajectory(np.linspace(token["start_time"] - WORD_TIMESTAMP_MARGIN,
                                                     token["end_time"] + WORD_TIMESTAMP_MARGIN, 10))
            plt.plot(sub_x, sub_y, linestyle='-', marker='o')
            plt.text(np.mean(sub_x), np.mean(sub_y), token["word"])
        plt.show()

    prob_dim = 24
    labels = {}

    for token, (_, pos) in zip(timestamps, pos_tags):
        if not pos.startswith("N") and not pos.startswith("J") and not pos.startswith("A"):
            continue
        word = token["word"].lower()
        if word in exclude_words:
            continue
        sub_x, sub_y, sub_widths = trajectory(
            np.linspace(token["start_time"] - WORD_TIMESTAMP_MARGIN,
                        token["end_time"] + WORD_TIMESTAMP_MARGIN, 30))
        mask_intensities = np.hanning(len(sub_x))
        if word in labels:
            prob_mat = labels[word].mask
            strokes = labels[word].strokes
            centers = labels[word].centers
        else:
            prob_mat = np.zeros((prob_dim, prob_dim))
            strokes = []
            centers = []
        strokes.append([])

        for x, y, width, intensity in zip(sub_x, sub_y, sub_widths, mask_intensities):
            variance = width * (prob_dim / image_size[0] + prob_dim / image_size[1]) / 2.0
            prob_x = x * prob_dim / image_size[0]
            prob_y = y * prob_dim / image_size[1]
            if prob_x < 0 or prob_x >= prob_dim or prob_y < 0 or prob_y >= prob_dim: continue
            row_offsets = np.maximum(0, np.abs(np.arange(prob_dim) - prob_y) - 1)
            col_offsets = np.maximum(0, np.abs(np.arange(prob_dim) - prob_x) - 1)
            prob_mat += (np.exp(-0.5 * (row_offsets[:,np.newaxis] ** 2 / variance)) *
                         np.exp(-0.5 * (col_offsets[np.newaxis,:] ** 2 / variance))) * intensity
            strokes[-1].append((x, y, width))

        center = (np.sum(sub_x * sub_widths / np.sum(sub_widths)),
                  np.sum(sub_y * sub_widths / np.sum(sub_widths)))
        # Normalize
        labels[word] = LabeledMask(token["word"], prob_mat, centers + [center], strokes)

    label_list = []
    for label in labels.values():
        label_list.append(LabeledMask(label.label, label.mask / np.max(label.mask),
                                      label.centers, label.strokes))

    if DEBUG:
        plt.figure()
        for k, label in enumerate(label_list):
            plt.subplot(int(len(label_list) // 3) + 1, 3, k + 1)
            plt.imshow(label.mask)
            plt.title(label.label)
        plt.tight_layout()
        plt.show()

    return label_list
