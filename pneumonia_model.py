import tensorflow as tf
from tensorflow import keras 
from keras import layers, models
'''
We'll choose four convolutional layers with increasing filters!!!!! :D :D :D 
This will help the model learn low-level (textures, etc) and high-level features (like lung structures). ¯\(°_o)/¯
'''

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3, 3), activation='relu'),  #yay more conv layers (´･ω･`)?
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),  #More neurons
        layers.Dropout(0.6),

        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid') #Single neuron for binary classification (normal/pneumonia)
    ])
    return model