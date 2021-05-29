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


