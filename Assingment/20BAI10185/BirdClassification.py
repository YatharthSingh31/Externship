# Imporitng libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation,Dropout,Flatten,Dense , BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Importing dataset
train_dir = 'C:\Users\91703\Practice\Assignments\SmartBrige assignments\BirdClassification\train_data'
test_dir =  "C:\Users\91703\Practice\Assignments\SmartBrige assignments\BirdClassification\test_data"


# Initialize data generators
train_datagen = ImageDataGenerator(
    rescale = 1. /255,
    zoom_range = 0.2,
    horizontal_flip = True,
    validation_split= 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size =(150,150),
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   subset='training')

validation_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size =(150,150),
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   subset='validation')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                   target_size =(150,150),
                                                   batch_size=32,
                                                   class_mode='categorical')

# model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# training model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // validation_generator.batch_size,
                    epochs=50)


# Evaluate the model using test generator
loss, accuracy = model.evaluate(test_generator)
plt.plot(history.history['loss'],color = 'red')
plt.plot(history.history['accuracy'],color = 'green')