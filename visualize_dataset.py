import matplotlib.pyplot as plt #To display images as visual plots 
import cv2 #To load + process images 
import os #For files 
from config import YOUR_DATASET_PATH

dataset_path = YOUR_DATASET_PATH 

normal_folder = os.path.join(dataset_path, "train", "NORMAL")
pneumonia_folder = os.path.join(dataset_path, "train", "PNEUMONIA")

def display_combined_images(normal_folder, pneumonia_folder, num_images): 
    pneumonia_images = os.listdir(pneumonia_folder)[:num_images]
    normal_images = os.listdir(normal_folder)[:num_images]

    plt.figure(figsize=(15, 7)) #Choosing a size for our plot

    for i, img_name in enumerate(normal_images): #displaying NORMAL images on the first line
        img_path = os.path.join(normal_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(3, num_images, i+1)
        plt.imshow(img, cmap="gray")
        plt.title("NORMAL LUNGS")
        plt.axis("off")

    for i, img_name in enumerate(pneumonia_images): #displaying PNEUMONIA images on the second line
        img_path = os.path.join(pneumonia_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title("PNEUMONIA")
        plt.axis("off")

    plt.show()

display_combined_images(normal_folder, pneumonia_folder, 5)
