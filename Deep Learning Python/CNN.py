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
import tensorflow.keras.backend as K

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

#This section is very important and is how we import our images! Don't mess with it
from keras.src.legacy.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()
train_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Train', class_mode = 'categorical', batch_size = 16, target_size=(32,32), shuffle = False)

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

#This prepares the model for running and specifies what metrics we are aiming for
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', precision, recall, f1, specificity, matthews_correlation_coefficient])

#This is our new input, and will be how we feed into model.fit, the command that actually trains the model.
test_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Test', class_mode = 'categorical', batch_size = 16, target_size=(32,32))
valid_data_keras = datagen.flow_from_directory(directory = 'C:/Users/grayj/Desktop/DATASET/Validate', class_mode = 'categorical', batch_size = 16, target_size=(32,32))
log_dir1 = 'C:/Users/grayj/Desktop/DATASET/logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir1, histogram_freq=1)
#The command that trains the model
history = model.fit(train_data_keras, epochs=4, validation_data=valid_data_keras, callbacks= [tensorboard_callback])

#This sets the model to evaluate itself on the testing data and write to Tensorboard
model.predict(test_data_keras, callbacks = [tensorboard_callback])
