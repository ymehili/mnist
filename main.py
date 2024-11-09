import tensorflow as tf
from random import random
from neuron import *

# Load the dataset
(img_train, result_train), (img_test, result_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to [0, 1]
img_train, img_test = img_train / 255.0, img_test / 255.0

def main():
    input_neurons = [Neuron([(random() % 100) / 100], (random() % 100) / 100, relu) for _ in range(784)]
    hidden_neurons1 = [Neuron([(random() % 100) / 100 for _ in range(784)], (random() % 100) / 100, relu) for _ in range(16)]
    hidden_neurons2 = [Neuron([(random() % 100) / 100 for _ in range(16)], (random() % 100) / 100, relu) for _ in range(16)]
    output_neurons = [Neuron([(random() % 100) / 100 for _ in range(16)], (random() % 100) / 100, relu) for _ in range(10)]
    
    for (img, result) in zip(img_train, result_train):
        # Flatten the 28x28 image to a 784x1 vector
        inputs = img.flatten()

        # Activate the input layer
        inputs = [neuron.activate(inputs) for neuron in input_neurons]

        # Activate the first hidden layer
        inputs = [neuron.activate(inputs) for neuron in hidden_neurons1]

        # Activate the second hidden layer
        inputs = [neuron.activate(inputs) for neuron in hidden_neurons2]

        inputs = [neuron.activate(inputs) for neuron in output_neurons]

        outputs = softmax(inputs)

        print(outputs)
    
    return

if __name__ == "__main__":
    main()
