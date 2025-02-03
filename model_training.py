from pneumonia_model import build_model
import numpy as np 
from preprocess_dataset import train_images, train_labels, test_images, test_labels, val_images, val_labels
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor= 'val_accuracy',
    patience=2,
    restore_best_weights=True
)

#Build the model using the script from pneumonia_model.py
model = build_model()

#Compile the model
model.compile(
    optimizer= Adam(learning_rate=0.0002),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

#Display model archiecture 
model.summary()

#Train the model 
history = model.fit(
    train_images, train_labels,
    epochs=12,
    validation_data=(val_images, val_labels),
    callbacks=[early_stopping]
)

#Evaluate the model on our test set 
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy*100:2f}%")

if test_accuracy*100 > 85: 
    #Save the model 
    model.save("pneumonia_model.keras")
    print("Model saved successfully!")

#Plot training and validation accuracy 
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()