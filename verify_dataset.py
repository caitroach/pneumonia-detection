import os 

#Replace this with the path where your dataset is located!
dataset_path = r"C:\Users\Roach\Desktop\pneumonia_project\chest_xray" #Ignore backslashes

#We're going to iterate through each folder and sub-folder to count our images!
for folder in ["train", "test", "val"]: 
    folder_path = os.path.join(dataset_path, folder)
    print(f"--- {folder.upper()} ---") #Just for formatting
    for subfolder in os.listdir(folder_path): 
        subfolder_path = os.path.join(folder_path, subfolder)
        num_images = len(os.listdir(subfolder_path))
        print(f"{subfolder}: {num_images} images")
