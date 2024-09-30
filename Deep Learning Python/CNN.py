#Attempting to run this on your own system requires changing the directory paths to match your own system.
#Also, make sure your directories are set up correctly for datagen.flow_from_directory!
#Run test(3) first, and then test (2) to make sure your directories are set up.
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #This does something important probably
import keras 
import tensorflow as tf
from keras import layers, models
from matplotlib import pyplot as plt
import pandas as pd
import numpy


#This whole section is likely outdated and will be replaced later.
directory1 = 'C:/Users/grayj/Desktop/DATASET'
df= pd.read_csv('C:/Users/grayj/Desktop/DATASET/HAM10000_metadata.csv')
file_paths = df['lesion_id'].values 
labels = df['dx'].values
#This whole section is likely outdated and will be replaced later.

#This section is very important and is how we import our images! Don't mess with it
from keras.src.legacy.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
train_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Train', class_mode = 'categorical', batch_size = 2, target_size=(32,32), shuffle = False)
x, y = next(train_data_keras)

#This section will be deleted, as it just shows the first 16 images the above section gets
for i in range(0,15):
    image = x[i].astype(int)
    plt.imshow(image)
    plt.show()
'''
#This is our actual model. I don't know how it really works but don't mess with it
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

#This does something
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#This is our new input, and will be how we feed into model.fit, the command that actually trains the model.
test_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Test', class_mode = 'categorical', batch_size = 16, target_size=(32,32))
valid_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Validate', class_mode = 'categorical', batch_size = 16, target_size=(32,32))
test_csv = pd.read_csv('C:/Users/grayj/Desktop/DATASET/test.csv', names = ["dx"])
train_csv = pd.read_csv('C:/Users/grayj/Desktop/DATASET/train.csv', names = ["dx"])
valid_csv = pd.read_csv('C:/Users/grayj/Desktop/DATASET/valid.csv', names = ["dx"])

model.fit(train_data_keras, train_csv, header=None, names='dx')
'''