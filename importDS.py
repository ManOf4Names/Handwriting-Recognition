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

input_dir = "./CharacterSets/Training"
val_split = 0.2
img_height = img_width = 50
batch_size = 128 # Garrett's laptop runs out of memory at 3179 batches at 100 size 

train_ds = tf.keras.utils.image_dataset_from_directory(
  input_dir,
  labels='inferred',
  validation_split=val_split,
  color_mode="grayscale",
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size) #WTF does batch size affect????

val_ds = tf.keras.utils.image_dataset_from_directory(
  input_dir,
  labels='inferred',
  validation_split=val_split,
  color_mode="grayscale",
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#prints the class names (so we know our dataset is being read)
'''class_names = train_ds.class_names
print(class_names)'''

'''Shuffles the training dataset for each epoch
#currently
image_count = len(list(train_ds))
print(image_count)
train_ds = train_ds.shuffle(image_count, reshuffle_each_iteration=True)'''







'''#THIS CODE IS JUST FOR VISUALIZATION, IT CAN BE SAFELY REMOVED
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
'''

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 62

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20
)
