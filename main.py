import tensorflow as tf
from random import random
from neuron import *

# Load the dataset
(img_train, result_train), (img_test, result_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to [0, 1]
img_train, img_test = img_train / 255.0, img_test / 255.0

def main():
    input_neurons = [Neuron([(random() % 100) / 100], (random() % 100) / 100, sigmoid) for _ in range(784)]
    hidden_neurons1 = [Neuron([(random() % 100) / 100 for _ in range(784)], (random() % 100) / 100, sigmoid) for _ in range(16)]
    hidden_neurons2 = [Neuron([(random() % 100) / 100 for _ in range(16)], (random() % 100) / 100, sigmoid) for _ in range(16)]
    output_neurons = [Neuron([(random() % 100) / 100 for _ in range(16)], (random() % 100) / 100, sigmoid) for _ in range(10)]
    
    for index, (img, result) in enumerate(zip(img_train, result_train)):
        # Flatten the 28x28 image to a 784x1 vector
        inputs = img.flatten()

        # Activate the input layer
        inputs = [neuron.activate(inputs) for neuron in input_neurons]

        # Activate the first hidden layer
        inputs = [neuron.activate(inputs) for neuron in hidden_neurons1]

        # Activate the second hidden layer
        inputs = [neuron.activate(inputs) for neuron in hidden_neurons2]

        # Activate the output layer
        inputs = [neuron.activate(inputs) for neuron in output_neurons]

        # Apply softmax to get the final output probabilities
        outputs = softmax(inputs)

        # Convert the result to a one-hot encoded vector
        label = [0] * 10
        label[result] = 1

        # Calculate the cross-entropy loss
        loss = cross_entropy_loss(outputs, label)

        max_output = max(outputs)
        max_index = outputs.index(max_output)

        print(f"Max output: {max_output}, Index: {max_index}, Loss: {loss}")
        
        if index == 10:
            break
    return

if __name__ == "__main__":
    main()
