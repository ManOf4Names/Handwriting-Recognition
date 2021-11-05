import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

#checks if the directory for our dataset exists,
if not os.path.exists("DS"):
    os.makedirs("DS")




image_data_generator = ImageDataGenerator() 


image = tf.keras.preprocessing.image.DirectoryIterator(
    "./CharacterSets", image_data_generator, target_size=(128, 128),
    color_mode='rgb', classes=None, class_mode='categorical',
    batch_size=100, shuffle=True, seed=None, data_format=None, save_to_dir="./DS",
    save_prefix='', save_format='png', follow_links=False,
    subset=None, interpolation='nearest', dtype=None
)

image.next() # This line will trigger the execution of Iterator. 