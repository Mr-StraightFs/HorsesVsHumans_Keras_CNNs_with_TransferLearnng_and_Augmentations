import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Load and prepare the inbuit Data (MNIST Fashion) from Keras
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#Visualize the Data : an array image from the dataset
plt.imshow(training_images[29])
print(training_labels[29])
print(training_images[29])

#Normalize Data
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Model Specification

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
# Loss and Gradient Parameters
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model Training
model.fit(training_images, training_labels, epochs=5)

