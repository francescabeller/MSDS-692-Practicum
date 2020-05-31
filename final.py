# Import packages
import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'

import cv2     # for capturing videos
import math   # for mathematical operations
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from keras.applications.vgg16 import preprocess_input


#####################################################################
############ READ VIDEO, EXTRACT FRAMES, SAVE AS IMAGES #############
#####################################################################

cd = os.getcwd()
os.mkdir(f'{cd}/frames')
# Extract pass video frames
for i in range(1,6):
    count = 0
    videoFile = f"{cd}\Videos\pass{i}.mp4"
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            ext = f'pass{i}_'
            filename =f"{cd}/frames/{ext}frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()
    print ("Done!")

# Extract run video frames
for i in range(1,6):
    count = 0
    videoFile = f"{cd}\Videos\\run{i}.mp4"
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = cap.get(cv2.CAP_PROP_FPS) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            ext = f'run{i}_'
            filename =f"{cd}/frames/{ext}frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, frame)
    cap.release()
    print("Done!")


########################################
############ IMAGE MAPPING #############
########################################

# Read in mapping.csv
data = pd.read_csv(f'{cd}/mapping.csv')

# Create array of image files
X = [ ]     # creating an empty array
for img_name in data.Image_ID:
    img = plt.imread('' + img_name)
    X.append(img)  # storing each image in array X
X = np.array(X)    # converting list to array

# Categorical encoding
y = data.Class
dummy_y = np_utils.to_categorical(y)    # one hot encoding Classes

# Reshape images
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)

# Pre-processing
X = preprocess_input(X, mode='tf')


######################################
############ BUILD MODEL #############
######################################

# Split training and test
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.25, random_state=42)

# Import packages
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout

# Load base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) # include_top=False to remove the top layer

# Make predictions, get features, retrain
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape

# Reshape to 1-D
X_train = X_train.reshape(70, 7*7*512)
X_valid = X_valid.reshape(24, 7*7*512)

# Pre-process images, make zero-centered
train = X_train/X_train.max()
X_valid = X_valid/X_train.max()


# Build the model
model = Sequential()
model.add(InputLayer((7*7*512,))) # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(2, activation='softmax')) # output layer

# Model summary
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))


#####################################
############ TEST MODEL #############
#####################################

# Install packages
from glob import glob
from tqdm import tqdm
from scipy import stats as s

# Get list of test file names
test_file_path = cd + r'\test_videos'
f = open(test_file_path + r'\\' + 'test_list.txt', 'r')
temp = f.read()
videos = temp.split('\n')

# Create dataframe
test = pd.DataFrame({'video_name': videos})
test_videos = test['video_name']

# Create lists to store tags
predict = []
actual = [1, 1, 1, 0, 0, 0]

# Create directory to hold test video frames
os.mkdir(f'{cd}/test_frames')

# Loop to extract test video frames
for i in tqdm(range(test_videos.shape[0])):
    count = 0
    videoFile = test_videos[i]
    cap = cv2.VideoCapture('test_videos/' + videoFile)
    frameRate = cap.get(5)
    x = 1
    '''
    # removing all other files from the temp folder
    files = glob('temp/*')
    for f in files:
        os.remove(f)
    '''
    while cap.isOpened():
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if frameId % math.floor(frameRate) == 0:
            # Store video frames in newly created 'test_frames' directory
            filename = 'test_frames/' + videoFile[:-4] + "_frame%d.jpg" % count
            count += 1
            cv2.imwrite(filename, frame)
    cap.release()

    # Read frames from 'test_frames' directory
    images = glob("test_frames/*.jpg")

    prediction_images = []
    for i in range(len(images)):
        img = image.load_img(images[i], target_size=(224, 224, 3))
        img = image.img_to_array(img)
        img = img / 255
        prediction_images.append(img)

    # converting all the frames for a test video into numpy array
    prediction_images = np.array(prediction_images)
    # extracting features using pre-trained model
    prediction_images = base_model.predict(prediction_images)
    # converting features in one dimensional array
    prediction_images = prediction_images.reshape(prediction_images.shape[0], 7 * 7 * 512)
    # predicting tags for each array
    prediction = model.predict_classes(prediction_images)
    # appending the mode of predictions in predict list to assign the tag to the video
    predict.append(y.columns.values[s.mode(prediction)[0][0]])
    # appending the actual tag of the video
    actual.append(videoFile.split('/')[1].split('_')[1])



