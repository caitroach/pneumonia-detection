'''
To give our model a chance, we need to pre-process our data!
We'll start by resizing every picture to a fixed size. 
Then we'll normalize pixel values to get smaller ranges, which are easier to work with.
'''

import numpy as np 
import cv2
import os
from config import YOUR_DATASET_PATH

dataset_path = YOUR_DATASET_PATH

'''
This function loads, resizes, and normalizes a single image!
'''

def preprocess_image(image_path, img_size=128):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #Load
    img = cv2.resize(img, (img_size, img_size)) #Resize
    img = img / 255.0 #Normalize
    return img

'''
This function iterates through our dataset and preprocesses each image.
'''

def preprocess_label_images(folder_path, img_size=128):
    images = []
    labels = []
    for label, subfolder in enumerate(["NORMAL", "PNEUMONIA"]):
        subfolder_path = os.path.join(folder_path, subfolder)
        for img_name in os.listdir(subfolder_path): 
            img_path = os.path.join(subfolder_path, img_name)
            img = preprocess_image(img_path, img_size)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

'''
Now we're applying all of that to our training, test, and validation data. 
'''

train_folder = os.path.join(dataset_path, "train")
test_folder = os.path.join(dataset_path, "test")
val_folder = os.path.join(dataset_path, "test")

train_images, train_labels = preprocess_label_images(train_folder, img_size=128)
test_images, test_labels = preprocess_label_images(test_folder, img_size=128)
val_images, val_labels = preprocess_label_images(val_folder, img_size=128)

print("Training data shape: ", train_images.shape, train_labels.shape)
print("Test data shape: ", test_images.shape, test_labels.shape)
print("Validataion data shape: ", val_images.shape, val_labels.shape)

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)
val_images = np.expand_dims(val_images, axis=-1)

print("Training data reshaped:", train_images.shape)
print("Test data reshaped:", test_images.shape)
print("Validation data reshaped:", val_images.shape)

