# MSDS-692-Practicum

## NFL Video Classification with Keras
Keras is one of the most used Python libraries for deep learning, and with its built-in, pre-trained video classification algorithms, it is a great tool for video classification. In this repository, code will be shown that generates a binary classification system for labeling NFL play videos as either a pass play or a run play using Keras' VGG16 algorithm.

## Pre-Code Work

Before coding could be used, I had to manually pull NFL videos from the web and cut these videos to create lists of pass play and run play videos to be used for training and testing. These videos are saved in this GitHub repository, and were also held in my computer's personal file system to be worked with in the Python code using the `os` package.

## Code

#### Import libraries

These are all of the packages necessary for creating the necessary dataframes, generating individual frames from input videos, managing file systems, and building/evaluating the VGG16 model.

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

#### Read video, extract frames, save as images

The below code was used to pull the training videos from the local GitHub file system, loop through each run/pass video, and extract individual frames to be saved in the newly created 'frames' folder. This is done with the use of the `cv2` Python package, which is an image and video processing library.

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

#### Image mapping

This section of code takes in `mapping.csv`, a manually created mapping file that includes the names of all the individual frames from the training video set, along with a binary 1/0 classification based on whether the frame includes passing elements (1) or not (0). The code then reads the images into an array `X`, assigns categorical dummies based on the mapping file, reshapes them to a uniform 224x224 shape, and preprocesses them using Keras' `preprocess_input`.

```python
# Read in mapping.csv
data = pd.read_csv(f'{cd}/mapping2.csv')

# Create array of image files
X = []
for img_name in data.Image_ID:
    img = plt.imread('' + img_name)
    X.append(img)
X = np.array(X)

# Categorical encoding
y = data.Class
dummy_y = np_utils.to_categorical(y)
y = y.to_frame()

# Reshape images
image = []
for i in range(0, X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224, 224)).astype(int)
    image.append(a)
X = np.array(image)

# Pre-processing
X = preprocess_input(X, mode='tf')
```

#### Model building

This section is where we actually build our VGG16 model. First, training and validation sets are created using the `train_test_split` function from Python's Scikit-Learn machine learning module with a 75/25 split. From there, an initial `base_model` is generated using Keras' `VGG16`, with weights taken from the pre-trained 'imagenet' and an input shape equivalent to the reshaping we performed in the previous section of code. Next, predictions are made on `X_train` and `X_valid` in order to get the dimensions needed as input for reshaping `X_train` and `X_valid` to 1D. Then, once again, we preprocess images to make them zero-centered, which then allows us to build and compile our model.

```python
# Split training and test
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.25, random_state=42)

# Load base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Make predictions, get features, retrain
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape

# Reshape to 1-D
X_train = X_train.reshape(275, 7 * 7 * 512)
X_valid = X_valid.reshape(92, 7 * 7 * 512)

# Pre-process images, make zero-centered
train = X_train / X_train.max()
X_valid = X_valid / X_train.max()

# Build the model
model = Sequential()
model.add(InputLayer((7 * 7 * 512,)))
model.add(Dense(units=1024, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

# Model summary
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))
```
