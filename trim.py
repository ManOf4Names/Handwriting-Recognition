from PIL import Image
import os.path
import sys

path = "D:\Handwriting-Recognition\CharacterSets\TrainingAlt\Test"
dirs = os.listdir(path)


def crop():
    for item in dirs:
        fullpath = os.path.join(path, item)  # corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((14, 14, 114, 114))  # corrected
            imCrop.save(f + 'Cropped.png', "png", quality=100)

crop()
