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
    
    epochs = 10
    learning_rate = 0.01

    for index, (img, result) in enumerate(zip(img_train, result_train)):
        inputs = img.flatten()

        # Forward pass
        input_activations = [neuron.activate(inputs) for neuron in input_neurons]
        hidden_activations1 = [neuron.activate(input_activations) for neuron in hidden_neurons1]
        hidden_activations2 = [neuron.activate(hidden_activations1) for neuron in hidden_neurons2]
        output_activations = [neuron.activate(hidden_activations2) for neuron in output_neurons]

        outputs = softmax(output_activations)

        label = [0] * 10
        label[result] = 1

        loss = cross_entropy_loss(outputs, label)

        # Backpropagation
        # Compute output layer error
        output_errors = [output - label for output, label in zip(outputs, label)]

        # Compute gradients for output layer
        for i, neuron in enumerate(output_neurons):
            for j in range(len(neuron.weights)):
                neuron.weights[j] -= learning_rate * output_errors[i] * hidden_activations2[j]
            neuron.bias -= learning_rate * output_errors[i]

        # Compute hidden layer 2 error
        hidden_errors2 = [sum(output_errors[k] * output_neurons[k].weights[i] for k in range(len(output_neurons))) for i in range(len(hidden_neurons2))]

        # Compute gradients for hidden layer 2
        for i, neuron in enumerate(hidden_neurons2):
            for j in range(len(neuron.weights)):
                neuron.weights[j] -= learning_rate * hidden_errors2[i] * hidden_activations1[j]
            neuron.bias -= learning_rate * hidden_errors2[i]

        # Compute hidden layer 1 error
        hidden_errors1 = [sum(hidden_errors2[k] * hidden_neurons2[k].weights[i] for k in range(len(hidden_neurons2))) for i in range(len(hidden_neurons1))]

        # Compute gradients for hidden layer 1
        for i, neuron in enumerate(hidden_neurons1):
            for j in range(len(neuron.weights)):
                neuron.weights[j] -= learning_rate * hidden_errors1[i] * input_activations[j]
            neuron.bias -= learning_rate * hidden_errors1[i]

    # Evaluate the model on the test dataset (to be implemented)
    # ...

if __name__ == "__main__":
    main()
