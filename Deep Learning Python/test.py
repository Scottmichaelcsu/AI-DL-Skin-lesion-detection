import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras 
import tensorflow as tf
from tensorflow.keras import layers, models
from keras import layers
import pandas as pd



directory = 'C:/Users/grayj/Desktop/DATASET'
df= pd.read_csv(directory + '/HAM10000_metadata.csv')
file_paths = df['lesion_id'].values
labels = df['dx'].values
#print(labels)
#ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))

def read_image(image_file, label):
    image = tf.io.read_file(directory + image_file)
    image = tf.image.decode_image(image, channels = 3, dtype = tf.float32)
    return image, label



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


history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# TEST TEST TEST changing this changing this