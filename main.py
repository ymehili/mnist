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

    for epoch in range(epochs):
        total_loss = 0
        for index, (img, result) in enumerate(zip(img_train, result_train)):
            # Flatten the 28x28 image to a 784x1 vector
            inputs = img.flatten()

            # Forward pass
            inputs = [neuron.activate(inputs) for neuron in input_neurons]
            hidden1_outputs = [neuron.activate(inputs) for neuron in hidden_neurons1]
            hidden2_outputs = [neuron.activate(hidden1_outputs) for neuron in hidden_neurons2]
            outputs = [neuron.activate(hidden2_outputs) for neuron in output_neurons]

            # Apply softmax to get the final output probabilities
            outputs = softmax(outputs)

            # Convert the result to a one-hot encoded vector
            label = [0] * 10
            label[result] = 1

            # Calculate the cross-entropy loss
            loss = cross_entropy_loss(outputs, label)
            total_loss += loss

            # Backpropagation
            # Calculate output layer gradients
            output_deltas = [(output - label[i]) for i, output in enumerate(outputs)]

            # Update output layer weights and biases
            for i, neuron in enumerate(output_neurons):
                for j in range(len(neuron.weights)):
                    neuron.weights[j] -= learning_rate * output_deltas[i] * hidden2_outputs[j]
                neuron.bias -= learning_rate * output_deltas[i]

            # Calculate hidden layer 2 gradients
            hidden2_deltas = [sum(output_deltas[k] * output_neurons[k].weights[i] for k in range(len(output_neurons))) * sigmoid(hidden2_outputs[i]) * (1 - sigmoid(hidden2_outputs[i])) for i in range(len(hidden_neurons2))]

            # Update hidden layer 2 weights and biases
            for i, neuron in enumerate(hidden_neurons2):
                for j in range(len(neuron.weights)):
                    neuron.weights[j] -= learning_rate * hidden2_deltas[i] * hidden1_outputs[j]
                neuron.bias -= learning_rate * hidden2_deltas[i]

            # Calculate hidden layer 1 gradients
            hidden1_deltas = [sum(hidden2_deltas[k] * hidden_neurons2[k].weights[i] for k in range(len(hidden_neurons2))) * sigmoid(hidden1_outputs[i]) * (1 - sigmoid(hidden1_outputs[i])) for i in range(len(hidden_neurons1))]

            # Update hidden layer 1 weights and biases
            for i, neuron in enumerate(hidden_neurons1):
                for j in range(len(neuron.weights)):
                    neuron.weights[j] -= learning_rate * hidden1_deltas[i] * inputs[j]
                neuron.bias -= learning_rate * hidden1_deltas[i]

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(img_train)}")

    # Evaluate the model on the test dataset
    correct_predictions = 0
    for img, result in zip(img_test, result_test):
        inputs = img.flatten()
        inputs = [neuron.activate(inputs) for neuron in input_neurons]
        hidden1_outputs = [neuron.activate(inputs) for neuron in hidden_neurons1]
        hidden2_outputs = [neuron.activate(hidden1_outputs) for neuron in hidden_neurons2]
        outputs = [neuron.activate(hidden2_outputs) for neuron in output_neurons]
        outputs = softmax(outputs)
        if outputs.index(max(outputs)) == result:
            correct_predictions += 1

    accuracy = correct_predictions / len(img_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
