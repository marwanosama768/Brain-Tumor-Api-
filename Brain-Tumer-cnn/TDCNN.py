import cv2
import os
import tensorflow as tf
from tensorflow import keras
from keras.utils import normalize
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Dense, Flatten, Activation

# Importing The Data
dimsize = 64
dataset = []
label = []
img_dir = "C:\\Users\\Ahmed\\grad project\\Tumor Detection using CNN\\Dataset"

not_imgs = os.listdir(img_dir + '\\no')
yest_imgs = os.listdir(img_dir + '\\yes')

for i, no_image in enumerate(not_imgs):
    if(no_image . split ( "." )[1] == "jpg"):
        image = cv2.imread(img_dir + "\\no\\" + no_image)
        image = Image.fromarray(image , "RGB")
        image = image.resize((dimsize, dimsize))
        dataset.append(image)
        label.append(0)

for i, yes_image in enumerate(yest_imgs):
    if(yes_image.split(".") [1] == "jpg"):
        image = cv2.imread(img_dir + "\\yes\\" + yes_image)
        image = Image.fromarray(image , "RGB")
        image = image.resize((dimsize, dimsize))
        dataset.append(image)
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size= 0.2, random_state=0)

print(x_train.shape)

# Normalize the data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# building the model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(dimsize, dimsize, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add((Activation('sigmoid')))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16,
          verbose=1,epochs=10,
          validation_data=(x_test, y_test), shuffle=False)

model.save('BTDCNN.h5')


