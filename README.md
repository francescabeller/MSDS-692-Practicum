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
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
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
from sklearn.metrics import accuracy_score, confusion_matrix
```

#### Read video, extract frames, save as images

The below code was used to pull the training videos from the local GitHub file system, loop through each run/pass video, and extract individual frames to be saved in the newly created 'frames' folder. This is done with the use of the `cv2` Python package, which is an image and video processing library.

```python
cd = os.getcwd()
os.mkdir(f'{cd}/frames')
# Extract pass video frames
for i in range(1, 21):
    count = 0
    videoFile = f"{cd}/training_videos/pass{i}.mp4"
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
            filename = f"{cd}/frames/{ext}frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()
    print(f"Done extracting pass{i}")

# Extract run video frames
for i in range(1, 21):
    count = 0
    videoFile = f"{cd}/training_videos/run{i}.mp4"
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
            filename = f"{cd}/frames/{ext}frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()
    print(f"Done extracting run{i}")
```

#### Image mapping

This section of code takes in `mapping.csv`, a manually created mapping file that includes the names of all the individual frames from the training video set, along with a binary 1/0 classification based on whether the frame includes passing elements (1) or not (0). The code then reads the images into an array `X`, assigns categorical dummies based on the mapping file, reshapes them to a uniform 224x224 shape, and preprocesses them using Keras' `preprocess_input`.

```python
# Read in mapping.csv
data = pd.read_csv(f'{cd}/mapping.csv')

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

#### Model testing/evaluation

This last section of code starts by extracting the `test_list.txt` file from the `test_videos` folder to get the full list of test video names. From there, a dataframe is created using the names and assigning class tags to them. Two lists are then created, one an empty list as a placeholder for the predictions our model will make, and the other a list of the true tags for our test videos. 

Before we can test, we must create a new directory to hold the individual frames that will be extracted from the test videos (`test_frames`). Each video file is looped through using a similar technique to what we did for the original training videos to extract the frames. Once these frames are all compiled in the `test_frames` folder, we use our trained VGG16 model to generate predictions. Finally, using `accuracy_score` and `confusion_matrix` from the scikit-learn Python library, we are able to determine the accuracy of our model predictions.

```python
# Get list of test file names
test_file_path = cd + r'\test_videos'
f = open(test_file_path + '\\' + 'test_list.txt', 'r')
temp = f.read()
videos = temp.split('\n')

# Create dataframe
test = pd.DataFrame({'video_name': videos})
test_videos = test['video_name']

classes = []
for index, row in test.iterrows():
    if 'pass' in row['video_name']:
        classes.append(1)
    else:
        classes.append(0)
test['class'] = classes

# creating the tags
test_y = test['class']
test_y = pd.get_dummies(test_y)

# Create lists to store tags
predict = []
actual = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# Create directory to hold test video frames
os.mkdir(f'{cd}/test_frames')

# Loop to extract test video frames
for i in tqdm(range(test_videos.shape[0])):
    count = 0
    video_file = test_videos[i]
    cap = cv2.VideoCapture('test_videos/' + video_file)
    frameRate = cap.get(5)
    x = 1

    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            # Store video frames in newly created 'test_frames' directory
            filename = 'test_frames/' + video_file[:-4] + "_frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()

    # Read frames from 'test_frames' directory
    images = glob("test_frames/*.jpg")

    prediction_images = []
    for i in range(len(images)):
        img = keras.preprocessing.image.load_img(images[i], target_size=(224, 224, 3))
        img = keras.preprocessing.image.img_to_array(img)
        img = img / 255
        prediction_images.append(img)

    # Convert test video frames to numpy array
    prediction_images = np.array(prediction_images)
    # Extract features using VGG16 model
    prediction_images = base_model.predict(prediction_images)
    # Convert extracted features to 1D array
    prediction_images = prediction_images.reshape(prediction_images.shape[0], 7 * 7 * 512)
    # Generate predictions
    prediction = model.predict_classes(prediction_images)
    # Append predictions to list
    predict.append(test_y.columns.values[s.mode(prediction)[0][0]])

# Get accuracy
accuracy_score(predict, actual) * 100
confusion_matrix(actual, predict)
```

## References
https://www.analyticsvidhya.com/blog/2019/09/step-by-step-deep-learning-tutorial-video-classification-python/
https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/
