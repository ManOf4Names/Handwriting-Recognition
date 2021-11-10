import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator 


from som import SOM

#checks if the directory for our dataset exists,
#if not os.path.exists("DS"):
#    os.makedirs("DS")




train_ds = tf.keras.utils.image_dataset_from_directory(
  "../CharacterSetsNew/CharacterSets/Training",
  labels='inferred',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(128, 128),
  batch_size=32)


class_names = train_ds.class_names
print(class_names)




#THIS CODE IS JUST FOR VISUALIZATION, IT CAN BE SAFELY REMOVED
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
#END OF VISUALIZATION CODE