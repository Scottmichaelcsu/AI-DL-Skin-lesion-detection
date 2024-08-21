import sys
import keras 
import tensorflow as tf
from keras import layers

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),

    ]
)