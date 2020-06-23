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

#####################################################################
############ READ VIDEO, EXTRACT FRAMES, SAVE AS IMAGES #############
#####################################################################

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

########################################
############ IMAGE MAPPING #############
########################################

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

######################################
############ BUILD MODEL #############
######################################

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

#####################################
############ TEST MODEL #############
#####################################

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
