# Pneumonia Detection With a Convolutional Neural Network ![Python](https://img.shields.io/badge/python-3.12.1-blue) ![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)

Built using Keras and trained on a dataset of 5,000+ X-ray images, this project is a Convolutional Neural Network (CNN) trained to detect viral and bacterial pneumonia in chest X-rays with a test accuracy of ~90%. 
Get ready for the longest README you've ever seen...

## Table of contents
- [Why pneumonia?](https://github.com/caitroach/pneumonia-detection/#why-pneumonia)
- [What is a convolutional neural network?](https://github.com/caitroach/pneumonia-detection/#what-is-a-convolutional-neural-network)
- [Project components](https://github.com/caitroach/pneumonia-detection/#project-components)
- [Running the model](https://github.com/caitroach/pneumonia-detection/#running-the-model)
- [What's next?](https://github.com/caitroach/pneumonia-detection/#whats-next)
- [Acknowledgements](https://github.com/caitroach/pneumonia-detection/#acknowledgements)
- [Contributing](https://github.com/caitroach/pneumonia-detection/#contributing)
  
## Why pneumonia? 
Pneumonia is a form of inflammation caused by a bacterial, viral, or fungal infection of the lungs. It is the single largest infectious cause of death in children worldwide and is most prominent in regions with high air pollution. For my first ML project, I decided to target a serious global healthcare issue, exploring the applications of machine learning to real-world issues.

## What is a convolutional neural network?
Pop quiz! What shape is this? 🔺

...I'm going to assume you got that right. A triangle! Good job. But how did you know? 

It seems simple, but think about it: we aren't born knowing what triangles are. We learned in early childhood that this shape 🔺 is associated with the label "triangle", and we've seen thousands of triangles throughout our lives. By that process, we learned that triangles have key characteristics that other shapes do not. Even though triangles come in all shapes and colours, we can identify them through their key features. 

Machine learning is very similar! Here's how it works: 
- You collect training data (like lots of triangle shapes)
- You let the model learn key features (three sides, three angles, three vertices), generalizing from that data
- Like a proud parent, you watch as it runs free, classifying triangles and other shapes it's never seen before. Great success!

That's where the "neural" in "neural network" comes from - we're using tech to mimic how our brains process information. (Cool, right?) The "convolution" in "convolutional neural network" (CNN) means that our image goes through little filters, making it easier for our model to detect meaningful patterns in the data and inform its learning. Rather than looking at each pixel individually, the computer looks at small sections of the image at a time, combining information to make its guesses. This convolution calculation helps the program understand the images we're showing it. Here's a more comprehensive look at image classification using CNNs via [AlmaBetter](https://www.almabetter.com/bytes/articles/convolutional-neural-networks): 

![tweety](https://github.com/user-attachments/assets/a361b3c4-4875-4b49-a645-b8bfc24bb172)

I decided to use this form of image recognition to identify viral and bacterial pneumonia from x-ray images. Unfortunately, lung structures are usually a little more complex than triangles or Looney Tunes characters, so I needed to clean up and preprocess the data before model training.

## Project components 
I worked on this project sequentially. This took two weeks and six steps: 

### 1. Finding a dataset 
I found [my dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) on Kaggle, which had over 5,000 images split into train, validation, and test sets. Finding a high-quality dataset can be challenging - to train an accurate model, you need data that is optimally labelled, distributed, and organized. This particular dataset had a validation set with only eight images per class (NORMAL, PNEUMONIA), which led to issues in training like overfitting. I manually redistributed the images after downloading them (moving from training to validation), which reduced overfitting and improved accuracy.  

### 2. Verifying the dataset 
Always check your dataset out before using it! I used ```os``` to count each image in my dataset and output a list with the image distribution. This makes it easier to evaluate the quality of a dataset before trying to train a machine learning model on it. 

### 3. Visualizing the dataset 
This step is technically optional, but after verifying that you have a good dataset, it's a good idea to visualize your data to get an idea of what you are working with. In my program, I used ```os``` to look through file directories, ```cv2``` to load and process the images in my dataset (if you can guess what CV stands for here, you get a prize!), and ```matplotlib``` to visualize the data. The result was a 6x2 grid of images labelled either "NORMAL LUNGS" or "PNEUMONIA". This was a nice way to understand the differences between our classes, before we even start working with our model. In my dataset, I noticed that some images had vastly different sizes. Good to know. 🤔

![pic](https://github.com/user-attachments/assets/80e98b93-1c3b-4be3-ba94-fbdc97e24d5a)

### 4. Preprocessing the dataset
Next, I needed to preprocess the dataset, standardizing our X-ray images so the model could learn patterns efficiently. I resized each image, converted them all to grayscale, normalized pixel values, and used binary classification (0 = Normal, 1 = Pneumonia) to simplify training. By the end, all X-ray images were labelled 128x128 grayscale images, ready for the model.

### 5. Designing the model
There are many different ways to do this, but I wanted to make this from scratch. I used Keras (pronounced kinda like "carrots") - a popular Python library that allows us to make and test our beautiful model. Keras is built on TensorFlow, another open-source library for machine learning, except Keras makes this a little simpler because it provides a nice clean Python frontend for us to work in. 

I used a CNN architecture with 4 convolutional layers with by max-pooling followed by 2 dense layers with dropout, and one final dense layer at the end for binary classification. I tweaked dropout rates and used early stopping to prevent overfitting.

I outlined the exact structure of the model, defining its architecture in a function called ```build_model```. 

### 6. Training the model
I imported my ```build_model``` function from step 5, applying it to my organized and preprocessed dataset. By default, this program outputs a summary of the model architecture.
![modelarchitecture](https://github.com/user-attachments/assets/a04c85b0-5fc4-4718-bc03-947bcd24b466)

Through each epoch, it tracks metrics like loss, training accuracy, and validation accuracy. At the end of every epoch, it outputs a final test accuracy, showing us how the model performs on unseen data over time. This will fluctuate depending on your architecture, dataset, and the quality of your sacrifices to the TensorFlow gods. Here, it's around 85%, but with early stopping, it ended up around 89.5%.

![image](https://github.com/user-attachments/assets/42a8a16c-88a9-428f-943a-67d33ddc192b)

Finally, the program uses ```matplotlib``` to graph the validation and training accuracies over each epoch to check for inconsistencies like overfitting (memorizing the training data instead of learning from it).

![pic2](https://github.com/user-attachments/assets/39400871-d28a-488b-83da-409c435da40c)

I experimented with data augmentation, learning rate, optimizers, and dropout rates until I reached an accuracy I was happy with. I found that early stopping made a huge improvement in test accuracy because it prevented the model from memorizing its training data instead of generalizing.

## Running the model 
### 1. Install dependencies
Make sure you have Python installed, then install dependencies: 

```pip install tensorflow keras numpy matplotlib opencv-python```

(You may need to use ```pip3``` instead of ```pip``` depending on your Python version.)

### 2. Clone the repo 
```git clone https://github.com/caitroach/pneumonia-detection.git```

```cd pneumonia-detection```

### 3. Run the model 
To train the model, run: 

```python model_training.py ```

## What's next?
- I only talked about accuracy here, which is not sufficient for healthcare applications because of the potential for false positives - or worse, false negatives. So I plan on assessing the model on metrics like precision and recall, F-1 Score, and specificity. I plan on plotting a confusion matrix to better visualize the model's mistakes.
- Right now, I'm working on an interactive web app where users can upload a sample chest x-ray image for live predictions. Once I can figure out Streamlit (😔), I plan to integrate Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps to show why the model makes a particular guess, instead of running this as a "black-box". This will help visualize the regions of the x-ray that informed the model's decision, making its guess interpretable to the user.
- I might also explore transfer learning or merging datasets to improve overall performance. The x-rays I used for this project were taken from only one age group in only one region (see [acknowledgements](https://github.com/caitroach/pneumonia-detection/#acknowledgements)), so it's not as diverse as it could be. Because of this, the model might struggle to make predictions in other populations. 

I'll keep this repo updated. Stay tuned! :D 

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
I invite you to experiment with this code and improve the model's accuracy! Feel free to fork this repo and submit a pull request. If you have any questions (or you just want to show off), you can hit me up at roachc006@gmail.com or post on [the discussion board](https://github.com/caitroach/pneumonia-detection/discussions).
