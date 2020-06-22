# MSDS-692-Practicum

## NFL Video Classification with Keras
Keras is one of the most used Python libraries for deep learning, and with its built-in, pre-trained video classification algorithms, it is a great tool for video classification. In this repository, code will be shown that generates a binary classification system for labeling NFL play videos as either a pass play or a run play using Keras' VGG16 algorithm.

## Pre-Code Work

Before coding could be used, I had to manually pull NFL videos from the web and cut these videos to create lists of pass play and run play videos to be used for training and testing. These videos are saved in this GitHub repository, and were also held in my computer's personal file system to be worked with in the Python code using the `os` package.

## Code

#### Import libraries

```python
# Import packages
import os
import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt
import keras.preprocessing.image
import numpy as np
from keras.utils import np_utils
from skimage.transform import resize
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from glob import glob
from tqdm import tqdm
from scipy import stats as s
from sklearn.metrics import accuracy_score

os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
```

These are all of the packages necessary for creating the necessary dataframes, generating individual frames from input videos, managing file systems, and building/evaluating the VGG16 model.

#### Read video, extract frames, save as images

```python
cd = os.getcwd()
os.mkdir(f'{cd}/frames2')
# Extract pass video frames
for i in range(1, 21):
    count = 0
    videoFile = f"{cd}/training_videos2/pass{i}.mp4"
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5)
    x = 1
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            ext = f'pass{i}_'
            filename = f"{cd}/frames2/{ext}frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()

# Extract run video frames
for i in range(1, 21):
    count = 0
    videoFile = f"{cd}/training_videos2/run{i}.mp4"
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(cv2.CAP_PROP_FPS)
    x = 1
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            ext = f'run{i}_'
            filename = f"{cd}/frames2/{ext}frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()
```

The above code was used to pull the training videos from the local GitHub file system, loop through each run/pass video, and extract individual frames to be saved in the newly created 'frames' folder. This is done with the use of the `cv2` Python package, which is an image and video processing library.
