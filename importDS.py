import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator 


image_data_generator = ImageDataGenerator() 
labels = ["TrainLowerA", "TrainLowerB", "TrainLowerC", "TrainLowerD", "TrainLowerE",
          "TrainLowerF", "TrainLowerG", "TrainLowerH", "TrainLowerI", "TrainLowerJ",
          "TrainLowerK", "TrainLowerL", "TrainLowerM", "TrainLowerN", "TrainLowerO",
          "TrainLowerP", "TrainLowerQ", "TrainLowerR", "TrainLowerS", "TrainLowerT",
          "TrainLowerU", "TrainLowerV", "TrainLowerW", "TrainLowerX", "TrainLowerY",
          "TrainLowerZ", "TrainNumber0", "TrainNumber1", "TrainNumber2", "TrainNumber3",
          "TrainNumber4", "TrainNumber5", "TrainNumber6", "TrainNumber7", "TrainNumber8",
          "TrainNumber9", "TrainUpperA", "TrainUpperB", "TrainUpperC", "TrainUpperD",
          "TrainUpperE", "TrainUpperF", "TrainUpperG", "TrainUpperH", "TrainUpperI",
          "TrainUpperJ", "TrainUpperK", "TrainUpperL", "TrainUpperM", "TrainUpperN",
          "TrainUpperO", "TrainUpperP", "TrainUpperQ", "TrainUpperR", "TrainUpperS",
          "TrainUpperT", "TrainUpperU", "TrainUpperV", "TrainUpperW", "TrainUpperX",
          "TrainUpperY", "TrainUpperZ"]

image = tf.keras.preprocessing.image.DirectoryIterator(
    "./CharacterSets/Training", image_data_generator, target_size=(50, 50),
    color_mode='grayscale', classes=labels, class_mode='categorical',
    batch_size=100, shuffle=True, seed=None, data_format=None, save_to_dir="./DS",
    save_prefix='', save_format='png', follow_links=False,
    subset="training", interpolation='nearest', dtype=None
)

image.next() # This line will trigger the execution of Iterator. 
