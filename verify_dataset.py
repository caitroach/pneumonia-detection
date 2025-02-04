import os 
from config import YOUR_DATASET_PATH

dataset_path = YOUR_DATASET_PATH 

#We're going to iterate through each folder and sub-folder to count our images!
for folder in ["train", "test", "val"]: 
    folder_path = os.path.join(dataset_path, folder)
    print(f"--- {folder.upper()} ---") #Just for formatting
    for subfolder in os.listdir(folder_path): 
        subfolder_path = os.path.join(folder_path, subfolder)
        num_images = len(os.listdir(subfolder_path))
        print(f"{subfolder}: {num_images} images")
