import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras 
import tensorflow as tf
from keras import layers, models
from matplotlib import pyplot as plt
import pandas as pd
import numpy

directory1 = 'C:/Users/grayj/Desktop/DATASET'
df= pd.read_csv('C:/Users/grayj/Desktop/DATASET/HAM10000_metadata.csv')
file_paths = df['lesion_id'].values
labels = df['dx'].values

#Attempting to import images from dataset into legacy Keras ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
train_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Train', class_mode = 'categorical', batch_size = 16, target_size=(32,32))
x, y = next(train_data_keras)

for i in range(0,15):
    image = x[i].astype(int)
    plt.imshow(image)
    plt.show()

model = models.Sequential()
model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28,28,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


test_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Test', class_mode = 'categorical', batch_size = 16, target_size=(32,32))
train_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Train', class_mode = 'categorical', batch_size = 16, target_size=(32,32))
valid_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Validate', class_mode = 'categorical', batch_size = 16, target_size=(32,32))

history = model.fit(train_data_keras, labels, epochs=10, 
                    validation_data=(valid_data_keras, test_labels))