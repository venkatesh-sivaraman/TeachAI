import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
import skimage.io as io
import os
import matplotlib.pyplot as plt
import cv2
import pickle
from labeling import LabeledMask
from nltk.stem import WordNetLemmatizer
import PIL

INPUT_IMAGE_SIZE = (224, 224)
NO_LABEL = "N/A"

HOG_FEATURE_SIZE = 1764

MASK_SIZE = (24, 24)
TRAIN_SIZES = (6, 4, 3)

PREDICT_SIZES = (INPUT_IMAGE_SIZE[0] // 2, INPUT_IMAGE_SIZE[0] // 3, INPUT_IMAGE_SIZE[0] // 4)

lemmatizer = WordNetLemmatizer()

SUPERLABELS = {
    "golden retriever": set(['golden', 'retriever']),
    "pug": set(['pug']),
    "poodle": set(['poodle']),
    "bassett": set(['basset', 'bassett']),
    "chihuahua": set(['chihuahua'])
}

class ImagePatch:
    """A patch of an image with an optional label."""

    def __init__(self, file_path, patch, label=None, superlabel=None):
        self.file_path = file_path
        self.patch = patch
        self.label = label
        self.superlabel = superlabel
        self._feature_vec = None

    @property
    def feature_vector(self):
        if self._feature_vec is None:
            color_img = (self.patch * 255.).astype(np.uint8)
            gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
            equalized_img = cv2.equalizeHist(gray_img)

            winSize = (64,64)
            blockSize = (16,16)
            blockStride = (8,8)
            cellSize = (8,8)
            nbins = 9
            derivAperture = 1
            winSigma = 4.
            histogramNormType = 0
            L2HysThreshold = (0.2)
            gammaCorrection = 0
            nlevels = 64
            hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                  histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
            winStride = (8,8)
            padding = (8,8)
            locations = ((10,20),)
            hog_hist = hog.compute(equalized_img,winStride,padding,locations).flatten()

            lab_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2LAB)
            hist = cv2.calcHist([lab_img], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            self._feature_vec = np.concatenate([hog_hist, hist])

        return self._feature_vec


class ClassificationModel:
    def __init__(self, initial_data_path=None, image_base=None):
        """
        Initializes a classification model.

        initial_data_path: If not None, a directory path containing pickle files, where each
            pickle file is a dictionary containing a 'labels' key with a list of
            labeling.LabeledMask objects.
        image_base: the base path for images if initial_data_path is provided
        """
        self.patches = []

        if initial_data_path and image_base:
            for path in os.listdir(initial_data_path):
                if not path.endswith(".pkl"): continue
                image_path = os.path.join(image_base, path[:path.find("_2020")])
                with open(os.path.join(initial_data_path, path), "rb") as file:
                    example = pickle.load(file)
                self.add_training_point(image_path, example['labels'])
            print("Loaded {} training patches from initial data".format(len(self.patches)))

    def has_used_image(self, image_path):
        """Returns true if the given image path is already in the training set."""
        print("Checking for", os.path.basename(image_path), set([os.path.basename(p.file_path) for p in
                                                             self.patches]))
        return any(os.path.basename(p.file_path) == os.path.basename(image_path) for p in self.patches)

    def img_to_array(self, img):
        x = np.asarray(img, dtype='float32')
        return x

    def add_patches(self, file_path, labels, num_per_label=2, crop_mass=0.9, extend_ratio=0.2, no_label=5):
        """
        Generates labeled patches of the given image. Operates by finding the bounding box of the
        mask of the given image, then extract patches at various stretch values and sizes.

        file_path: A path to an image
        labels: a list of labeling.LabeledMask objects
        num_per_label: the number of patches per label word to add
        crop_mass: The proportion of the density of the mask to include in the bounding box
        extend_ratio: The max fraction of the bounding box dimensions by which to dilate the
            crop box for each patch (random).
        no_label: The number of patches with no label to add to the list.
        """
        original_image = PIL.Image.open(file_path)
        original_size = original_image.size
        patches = []
        used_regions = np.zeros(MASK_SIZE)

        superlabel = next((sl for sl, variants in SUPERLABELS.items()
                       if any(lemmatizer.lemmatize(l.label.lower()) in variants
                              for l in labels)), NO_LABEL)

        for label in labels:
            mask = label.mask
            mass = np.sum(np.where(np.isnan(mask), 0, mask))
            if mass <= 1:
                continue

            txt_label = lemmatizer.lemmatize(label.label.lower())

            # Make a cumulative histogram of the intensities of each row and column
            row_intensities = np.cumsum(mask.sum(axis=1))
            col_intensities = np.cumsum(mask.sum(axis=0))
            row_intensities /= row_intensities[-1]
            col_intensities /= col_intensities[-1]
            row_func = interp1d(row_intensities, np.arange(len(row_intensities)), fill_value=0.0, bounds_error=False)
            col_func = interp1d(col_intensities, np.arange(len(col_intensities)), fill_value=0.0, bounds_error=False)

            # determine the bounding rectangle that contains crop_mass of the mass on each axis
            row_min, row_max = row_func([(1 - crop_mass) / 2.0, (1 + crop_mass) / 2.0])
            col_min, col_max = col_func([(1 - crop_mass) / 2.0, (1 + crop_mass) / 2.0])
            height = row_max - row_min
            width = col_max - col_min
            used_regions[int(row_min):int(row_max), int(col_min):int(col_max)] = 1

            # now sample various crops of this box extended by at most extend_ratio
            for _ in range(num_per_label):
                top, left, bottom, right = np.random.uniform(-extend_ratio * height, extend_ratio * height, size=(4,))
                crop_box = (
                    max(col_min - left, 0) * original_size[0] / mask.shape[1],
                    max(row_min - top, 0) * original_size[0] / mask.shape[0],
                    min(col_max + right, mask.shape[1]) * original_size[0] / mask.shape[1],
                    min(row_max + bottom, mask.shape[0]) * original_size[0] / mask.shape[0],
                )
                cropped_image = original_image.crop(crop_box)
                cropped_arr = self.img_to_array(cropped_image.resize(INPUT_IMAGE_SIZE)) / 255.
                self.patches.append(ImagePatch(os.path.basename(file_path),
                                               cropped_arr, txt_label, superlabel))

        # Now sample unused regions of the image for a "no label" label
        num_no_labels = 0
        while num_no_labels < no_label:
            size = np.random.choice(TRAIN_SIZES)
            y = np.random.choice(used_regions.shape[0] - size)
            x = np.random.choice(used_regions.shape[1] - size)
            subregion = used_regions[x:x + size, y:y + size]
            if np.mean(subregion) > 0.05:
                continue
            crop_box = (
                x * original_size[0] / used_regions.shape[1],
                y * original_size[0] / used_regions.shape[0],
                (x + size) * original_size[0] / used_regions.shape[1],
                (y + size) * original_size[0] / used_regions.shape[0],
            )
            cropped_image = original_image.crop(crop_box)
            cropped_arr = self.img_to_array(cropped_image.resize(INPUT_IMAGE_SIZE)) / 255.
            patches.append(ImagePatch(os.path.basename(file_path), cropped_arr, NO_LABEL, NO_LABEL))
            num_no_labels += 1

        return patches, labels

    def add_training_point(self, image_path, labels):
        """
        Adds a training point to the model.

        image_path: Path to the image from which the labels were extracted.
        labels: List of labeling.LabeledMask objects representing the labels for the image.
        """
        self.add_patches(image_path, labels, num_per_label=5, no_label=5)

    def predict_patch(self, patch, k=5, plot=False):
        """
        Predicts labels and probabilities for the given ImagePatch.

        patch: An ImagePatch object
        k: the number of nearest neighbors to use to determine label scores

        Returns: (labels, superlabels)
            labels: a sorted list of (label, similarity) tuples.
            superlabels: the same but with superlabels.
        """
        features_X = np.stack([p.feature_vector for p in self.patches])
        query = patch.feature_vector.reshape(1, -1)

        # Generate normalized cosine similarities to each patch
        sims = (cosine_similarity(query[:,:HOG_FEATURE_SIZE], features_X[:,:HOG_FEATURE_SIZE])[0] * 0.2 +
                cosine_similarity(query[:,HOG_FEATURE_SIZE:], features_X[:,HOG_FEATURE_SIZE:])[0] * 0.8)
        sims /= np.max(sims)

        label_results = []
        superlabel_results = []
        seen_labels = set()
        seen_superlabels = set()
        for k, (i, sim) in enumerate(sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]):
            label = self.patches[i].label
            if label not in seen_labels and label != NO_LABEL:
                label_results.append((label, sim))
                seen_labels.add(label)
            superlabel = self.patches[i].superlabel
            if superlabel not in seen_superlabels and superlabel != NO_LABEL:
                superlabel_results.append((superlabel, sim))
                seen_superlabels.add(superlabel)

        if plot:
            plt.figure(figsize=(9, 4))
            plt.subplot(2, 5, 1)
            plt.imshow(patch.patch)

            for k, (i, sim) in enumerate(sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:9]):
                plt.subplot(2, 5, k + 2)
                plt.imshow(self.patches[i].patch)
                plt.title("{:.2f} - {}".format(sim, self.patches[i].label))
            plt.tight_layout()
            plt.show()

        return label_results, superlabel_results

    def predict_image(self, image_path):
        """
        Predicts the most likely labels for the given image, as well as masks indicating the
        regions of intensity for each label.

        image_path: Path to an image to predict.
        Returns: (superlabel, confidence, labels, masks, total_mask)
            superlabel: The most likely superlabel for this image
            confidence: The confidence (from 0-1) that the superlabel is correct
            labels: A sorted list of (label, similarity) tuples
            masks: A dictionary from label strings to mask arrays of size (INPUT_IMAGE_SIZE,
                INPUT_IMAGE_SIZE) where the value indicates the relevance of the label to the
                region from 0-1
            total_mask: A summed and normalized mask for the entire image.
        """

        image = PIL.Image.open(image_path)
        resized = image.resize(INPUT_IMAGE_SIZE)
        overall_probs = {}
        label_probs = {}
        masks_by_label = {}
        for size in PREDICT_SIZES:
            for y in range(0, INPUT_IMAGE_SIZE[0] - size, size // 2):
                for x in range(0, INPUT_IMAGE_SIZE[1] - size, size // 2):
                    subimage = resized.crop((x, y, x + size, y + size)).resize(INPUT_IMAGE_SIZE)
                    subarray = self.img_to_array(subimage) / 255.
                    patch = ImagePatch(os.path.basename(image_path), subarray)
                    probs, super_probs = self.predict_patch(patch)
                    for label, prob in super_probs:
                        overall_probs[label] = overall_probs.get(label, 0.0) + prob

                    for label, prob in probs:
                        label_probs[label] = label_probs.get(label, 0.0) + prob
                        # Update the mask for this label in the given region
                        mask = masks_by_label.setdefault(label, np.zeros(INPUT_IMAGE_SIZE))
                        prob_mask = np.hanning(size)[:,np.newaxis] * np.hanning(size)[np.newaxis,:] * prob
                        mask[x:x + size, y: y + size] += prob_mask

        total_mask = sum(list(masks_by_label.values())) ** 2
        total_mask /= np.max(total_mask)
        likely_superlabel = max(overall_probs, key=overall_probs.get)
        confidence = overall_probs[likely_superlabel] / sum(overall_probs.values())
        allowed_labels = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)
        allowed_labels = [(l, s) for l, s in allowed_labels if l not in
                          SUPERLABELS[likely_superlabel]]
        return (likely_superlabel,
                confidence,
                allowed_labels,
                {l: m / np.max(m) for l, m in masks_by_label.items()},
                total_mask)

