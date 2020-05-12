# TeachAI

TeachAI is a system for training image classification models using gesture and speech input. This implementation is built to recognize 5 breeds of dogs, but the concept could be extended to a variety of classification tasks, such as medical images.

## Requirements

* The project requires Python 3+ and is tested on MacOS (requires system speech capabilities, which may not work on other platforms). The following command should install all necessary dependencies:

```
pip install numpy scipy kivy Pillow scikit-learn opencv-python nltk
```

* A Leap Motion is required, and the Leap Motion SDK must be installed with the application active.
* Google Cloud Speech-to-Text credentials must be obtained and set in the current shell. Once you have downloaded a private API key from [Google Cloud](https://cloud.google.com/speech-to-text), the following command will activate the credentials:

```
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

## How to Run

```
python main.py
```

## Source Code Overview

The most important files in the project are described below:

### View Controllers

* `main.py`, `main.kv` - the master files for starting the application. The `MainWidget` class is responsible for showing dialogs, advancing the images and saving them, and managing the classification model.
* `start_view.py` - widget that shows when the application is first started.
* `image_controller.py` - widget that handles image annotation and synthesizes it with speech input from the microphone.
* `confirm_dialog.py` - widget that shows a confirmation of the labeling and allows the user to confirm or redo.

### UI Helpers

* `button.py` - a simple button that activates when the user hovers their hand over it for sufficient time.
* `speech.py` - a widget that displays an animated microphone and collects speech input, as well as transcribes it asynchronously.
* `system_speech.py` - wrapper around system speech call to MacOS.
* `paintbrush.py` - a widget that renders a variable-width, fading paintbrush.

### Computation Helpers

* `classification.py` - contains the `ClassificationModel` class, which implements the nearest-neighbor classifier.
* `labeling.py` - contains the `label_region` function, which generates a set of labeled masks given timestamped gestures and speech input.

