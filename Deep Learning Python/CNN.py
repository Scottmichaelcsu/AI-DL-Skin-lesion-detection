#Attempting to run this on your own system requires changing the directory paths to match your own system.
#Also, make sure your directories are set up correctly for datagen.flow_from_directory!
#Run test(3) first, and then test (2) to make sure your directories are set up, and then 2.5 to finalize file structure.
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #This does something important probably
import keras 
import tensorflow as tf
from keras import layers, models
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


#This whole section is likely outdated and will be replaced later.
directory1 = 'C:/Users/grayj/Desktop/DATASET'
df= pd.read_csv('C:/Users/grayj/Desktop/DATASET/HAM10000_metadata.csv')
file_paths = df['lesion_id'].values 
labels = df['dx'].values
#This whole section is likely outdated and will be replaced later.

#This section is very important and is how we import our images! Don't mess with it
from keras.src.legacy.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
train_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Train', class_mode = 'categorical', batch_size = 16, target_size=(32,32), shuffle = False)
x, y = next(train_data_keras)


#This is our actual model.
model = models.Sequential()
model.add(layers.Conv2D(16, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))

#This does something important don't mess with it 
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#This is our new input, and will be how we feed into model.fit, the command that actually trains the model.
test_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Test', class_mode = 'categorical', batch_size = 16, target_size=(32,32))
valid_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Validate', class_mode = 'categorical', batch_size = 16, target_size=(32,32))
test_csv = pd.read_csv('C:/Users/grayj/Desktop/DATASET/test.csv', names = ["dx"])
train_csv = pd.read_csv('C:/Users/grayj/Desktop/DATASET/train.csv', names = ["dx"])
valid_csv = pd.read_csv('C:/Users/grayj/Desktop/DATASET/valid.csv', names = ["dx"])

#The command that trains the model
history = model.fit(train_data_keras, epochs=1, validation_data=valid_data_keras)

#This should set the model to predict on the testing data- however it gives no result?
score = model.evaluate(test_data_keras)


#I tried to chart the data using plotlib according to that one guy's videos on HAM10000
#So far didn't work, but I might try another method.


#model.summary()