# TeachAI

TeachAI is a system for training image classification models using gesture and speech input. This implementation is built to recognize 5 breeds of dogs, but the concept could be extended to a variety of classification tasks, such as medical images.

## Requirements

* The project requires Python 3+ and is tested on MacOS (requires system speech capabilities, which may not work on other platforms). The following command should install all necessary dependencies:

```
pip install numpy scipy kivy Pillow scikit-learn opencv-python nltk
```

* A Leap Motion is required, and the Leap Motion SDK must be installed with the application active.
* Google Cloud Speech-to-Text credentials must be obtained and set in the current shell. Once you have downloaded a private API key from [https://cloud.google.com/speech-to-text](Google Cloud), the following command will activate the credentials:

```
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

## Running

```
python main.py
```
