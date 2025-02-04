# Pneumonia Detection With a Convolutional Neural Network
Built using Keras and trained on a dataset of 5,000+ X-ray images, this project is a Convolutional Neural Network (CNN) trained to detect viral and bacterial pneumonia in chest X-rays with a test accuracy of ~90%. 
Get ready for the longest README you've ever seen...

## Table of contents
- [Why pneumonia?](https://github.com/CaitlinRoach06/pneumonia-detection/#why-pneumonia)
- [What is a convolutional neural network?](https://github.com/CaitlinRoach06/pneumonia-detection/#what-is-a-convolutional-neural-network)
- [Project components](https://github.com/CaitlinRoach06/pneumonia-detection/#project-components)
- [Running the model](https://github.com/CaitlinRoach06/pneumonia-detection/#running-the-model)
- [What's next?](https://github.com/CaitlinRoach06/pneumonia-detection/#whats-next)
- [Acknowledgements](https://github.com/CaitlinRoach06/pneumonia-detection/#acknowledgements)
- [Contributing](https://github.com/CaitlinRoach06/pneumonia-detection/#contributing)
  
## Why pneumonia? 
Pneumonia is a form of inflammation caused by a bacterial, viral, or fungal infection of the lungs. It is the single largest infectious cause of death in children worldwide and is most prominent in regions with high air pollution. For my first machine learning project, I decided to target a serious global healthcare issue, exploring the applications of machine learning to real-world issues.

## What is a convolutional neural network?
Pop quiz! What shape is this? 🔺

...I'm going to assume you got that right. A triangle! Good job. But how did you know? 

It seems simple, but think about it: we aren't born knowing what triangles are. We learned in early childhood that this shape 🔺 is associated with the label "triangle", and we've seen thousands of triangles throughout our lives. By that process, we learned that triangles have key characteristics that other shapes do not. Even though triangles come in all shapes and colours, we can identify them through their key features. 

Machine learning is very much similar. If you wanted to teach your computer what a triangle is, you'd collect dozens of triangle shapes, associate them with the label TRIANGLE, and show them to your program one by one, testing how well it knows its shapes. Over time, much like a human, your computer learns the characteristics of the shape, and is able to generalize them to shapes it's never seen before. 

That's where the "neural" in "neural network" comes from - we're using tech to mimic how our brains process information. The "convolution" in "convolutional neural network" (CNN) means that our image goes through little filters, making it easier for our model to detect meaningful patterns in the data and inform its learning. Rather than looking at each pixel individually, the computer looks at small sections of the image at a time, combining information to make its guesses. This convolution calculation helps the program understand the images we're showing it. Here's a more comprehensive look at image classification using CNNs via [AlmaBetter](https://www.almabetter.com/bytes/articles/convolutional-neural-networks): 
![tweety](https://github.com/user-attachments/assets/a361b3c4-4875-4b49-a645-b8bfc24bb172)

I decided to use this form of image recognition to identify viral and bacterial pneumonia from x-ray images. Unfortunately, lung structures are a little more complex than triangles, so I needed to clean up and preprocess the data before model training.

## Project components 
I worked on this project sequentially. This took two weeks and six steps: 

### 1. Finding a dataset 
I found [my dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) on Kaggle, which had over 5,000 images split into train, validation, and test sets. . Finding a high-quality dataset can be challenging - to train an accurate model, you need data that is optimally labelled, distributed, and organized. This particular dataset had a validation set with only eight images per class (NORMAL, PNEUMONIA), which led to inefficiencies in training. I manually redistributed the images after downloading them (moving from training to validation), leading to higher test accuracy overall. 

### 2. Verifying the dataset 
It's good practice to check out the structure of your dataset before doing anything with it. I used ```os```, a Python module for interacting with the operating system, to count each image in my dataset and output a list with the image distribution. This makes it easier to evaluate the quality of a dataset before trying to train a machine learning model on it. 

### 3. Visualizing the dataset 
This step is technically optional, but after verifying that you have a good dataset, it's a good idea to visualize your data to get an idea of what you are working with. In my program, I used ```os``` to look through file directories, ```cv2``` to load and process the images in my dataset, and ```matplotlib``` to visualize the data. The result was a 6x2 grid of images labelled either "NORMAL LUNGS" or "PNEUMONIA". This was a nice way to understand the differences between our classes, before we even start working with our model. 
![pic](https://github.com/user-attachments/assets/80e98b93-1c3b-4be3-ba94-fbdc97e24d5a)

### 4. Preprocessing the dataset
Next, I needed to preprocess the dataset, standardizing our X-ray images so the model could learn patterns efficiently. I resized each image, converted them all to grayscale, normalized pixel values, and used binary classification (0 = Normal, 1 = Pneumonia) to simplify the classification. By the end, all X-ray images were labelled 128x128 grayscale images, ready for training.

### 5. Designing the model
There are many different ways to do this, but I wanted to make this from scratch. I used Keras (pronounced kinda like "carrots") - a popular Python library that allows us to make and test our beautiful model. Keras is built on TensorFlow, another open-source library for machine learning, except Keras makes this a little simpler because it provides a nice clean Python frontend for us to work in. 
I outlined the exact structure of the model, defining its architecture in a function called ```build_model```. 

### 6. Training the model
I imported my ```build_model``` function from step 5, applying it to my organized and preprocessed dataset. By default, this program outputs a summary of the model architecture.
![modelarchitecture](https://github.com/user-attachments/assets/a04c85b0-5fc4-4718-bc03-947bcd24b466)

Through each epoch, it tracks metrics like loss, training accuracy, and validation accuracy. At the end of every epoch, it outputs a final test accuracy, showing us how the model performs on unseen data over time. 
![image](https://github.com/user-attachments/assets/42a8a16c-88a9-428f-943a-67d33ddc192b)

Finally, it uses ```matplotlib``` to graph the validation and training accuracies over each epoch to check for overfitting (memorizing the training data instead of generalizing from it).
![pic2](https://github.com/user-attachments/assets/39400871-d28a-488b-83da-409c435da40c)

I experimented with data augmentation, learning rate, optimizers, and dropout rates until I reached an accuracy I was happy with. I found that early stopping made a huge improvement in test accuracy because it prevented the model from memorizing its training data.

## Running the model 
### 1. Install dependencies
Make sure you have Python installed, then install dependencies: 

```pip install tensorflow keras numpy matplotlib opencv-python```

(You may need to use ```pip3``` instead of ```pip``` depending on your Python version.)

### 2. Clone the repo 
```git clone https://github.com/CaitlinRoach06/pneumonia-detection.git```

```cd pneumonia-detection```

### 3. Run the model 
To train the model, run: 

```python model_training.py ```

## What's next?
I plan to improve the model's accuracy, possibly exploring transfer learning for higher efficiency and looking into more expansive datasets. I would love to deploy this project as a FastAPI Web App in the near future as well. I'll keep this repo updated! 

## Acknowledgements 
### Dataset 
You can find the dataset I used here: [Paul Mooney: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

These X-ray images were selected from cohorts of pediatric patients of one to five years old from Guangzhou Women and Children's Medical Center in Guangzhou, China. All imaging was performed as part of routine patient care.

### Resources
I referenced lots of tutorials, documentation, research papers, and YouTube videos to build this project. Here are some of the resources I used:  
- [GeeksForGeeks: Pneumonia Detection Using CNN in Python](https://www.geeksforgeeks.org/pneumonia-detection-using-cnn-in-python/)
- [D. Varshni, K. Thakral, L. Agarwal, R. Nijhawan and A. Mittal, "Pneumonia Detection Using CNN based Feature Extraction," 2019](https://ieeexplore.ieee.org/document/8869364)
- [Floxus: CREATE A PNEUMONIA DETECTION MODEL | Learning CNN techniques](https://www.youtube.com/watch?v=5nVWENPbas0)
- [An Q, Chen W, and Shao W., "A Deep Convolutional Neural Network for Pneumonia Detection in X-ray Images with Attention Ensemble," 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10887593/)

## Contributing
I invite you to experiment with this code and improve the model's accuracy! Feel free to fork this repo and submit a pull request. If you have any questions (or you just want to show off), you can hit me up at roachc006@gmail.com or open an issue here on GitHub.
