# Batch of imports
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from som import Kohonen


def condense_color(tensor):
    vect = np.array([])
    for i in tensor:
        for j in i:
            avg = np.average(j.numpy(), returned=True)[0]
            if avg > 0.5:
                vect = np.append(vect, int(0))
            else:
                vect = np.append(vect, int(1))
    return vect


def condense_gray(tensor):
    threshold = 0.5
    l = tensor.numpy().shape[0]
    w = tensor.numpy().shape[1]
    vect = tensor.numpy().reshape(-1,) < threshold
    # imgPrint(vect, l, w)
    return vect


def imgPrint(a, l, w):
    for i in range(l):
        line = ""
        for j in range(w):
            line += str(int(a[i * l + j])) + " "
        print(line)


input_dir = "./CharacterSets/Training"  # Training set directory
val_split = 0.2                         # Validation split
img_height = img_width = 50             # Height and width of the input images
batch_size = 100                        # Batch size

# Defining the training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    input_dir,
    labels='inferred',
    validation_split=val_split,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    color_mode="grayscale",
    batch_size=batch_size)  # WTF does batch size affect????

# Defining the validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    input_dir,
    labels='inferred',
    validation_split=val_split,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Normalization layer. first_image can be viewed as a numpy array, but currently consists of all 1s for some reason.
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_valid = val_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch_t, labels_batch_t = next(iter(normalized_train))
image_batch_v, labels_batch_v = next(iter(normalized_valid))

# Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))
trainSet = np.array([])
validSet = np.array([])
for i in range(len(image_batch_t)):
    trainSet = np.append(trainSet, condense_gray(image_batch_t[i]))
    validSet = np.append(trainSet, condense_gray(image_batch_v[i]))
trainSet = trainSet.reshape(-1, 2500)

som = Kohonen(2500, 100, net_dim=(50, 50), n_classes=62)
som.train(trainSet)

labelsValid = []
labelsTrain = []
for i in labels_batch_v:
    labelsValid.append(i.numpy())
for i in labels_batch_t:
    labelsTrain.append(i.numpy())
