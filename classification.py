import numpy as np
import skimage.io as io
import os
import keras
from keras import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dense, Softmax, Deconvolution2D, UpSampling2D, Input, LeakyReLU, Conv2DTranspose
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


class LabeledRegion:
    def __init__(self, mask, label, temporary=False):
        self.mask = mask
        self.label = label
        self.temporary = temporary

INPUT_IMAGE_SIZE = (224, 224)
MASK_SIZE = (14, 14)
NUM_LABELS = 10

class ClassificationModel:
    def __init__(self, weights_path):
        input_img = Input(shape=(*INPUT_IMAGE_SIZE, 3))

        x = Conv2D(12, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(36, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x) # not used! lol
        x = Conv2DTranspose(36, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = Conv2DTranspose(24, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = Conv2DTranspose(12, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoenc_model = Model(inputs=input_img, outputs=output)
        autoenc_model.load_weights(weights_path)

        classifier = Conv2D(NUM_LABELS, (3, 3), activation='softmax', padding='same')(encoded)
        self.model = Model(inputs=input_img, outputs=classifier)
        self.model.compile(optimizer="adam", loss="binary_crossentropy")
        print(self.model.summary())

        self.known_labels = {}
        self.train_X = []
        self.train_Y = []

    def add_training_example(self, image_path, regions):
        """
        image_path should be a path to a jpg image. regions should be a list of LabeledRegion
        objects, each containing a mask of size 14x14.
        """
        # Add the resized version of the image to train_X
        original = load_img(image_path, target_size=INPUT_IMAGE_SIZE)
        img_arr = img_to_array(original) / 255.
        self.train_X.append(img_arr)

        # Add a mask containing the regions to train_Y
        total_mask = np.zeros((*MASK_SIZE, NUM_LABELS))
        total_mask[:,:,0] = 1
        for region in regions:
            if region.label not in self.known_labels:
                print("Assigning position {} to label {}".format(len(self.known_labels) + 1,
                                                                 region.label))
                self.known_labels[region.label] = len(self.known_labels) + 1
            total_mask[region.mask != 0, 0] = 0
            total_mask[region.mask != 0, self.known_labels[region.label]] = 1
        total_mask /= np.sum(total_mask, axis=2, keepdims=True)
        self.train_Y.append(total_mask)

        X = np.stack(self.train_X, axis=0)
        Y = np.stack(self.train_Y, axis=0)
        self.model.fit(X, Y, epochs=5)

    def predict_training_example(self, image_path):
        original = load_img(image_path, target_size=INPUT_IMAGE_SIZE)
        img_arr = img_to_array(original) / 255.
        pred = self.model.predict(np.expand_dims(img_arr, axis=0))
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img_arr)
        plt.subplot(1, 2, 2)
        plt.imshow(pred[0,:,:,1])
        plt.colorbar()
        plt.show()
