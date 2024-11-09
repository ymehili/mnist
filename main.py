import tensorflow as tf
from neuron import *
# Load the dataset
(img_train, result_train), (img_test, result_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to [0, 1]
img_train, img_test = img_train / 255.0, img_test / 255.0


def main():
    for (img, result) in zip(img_train, result_train):
        input_neurons = [Neuron([0.0], 0.0, relu) for _ in range(784)]
        hidden_neurons1 = [Neuron([0.0 for _ in range(784)], 0.0, relu) for _ in range(128)]
        hidden_neurons2 = [Neuron([0.0 for _ in range(128)], 0.0, relu) for _ in range(64)]
        output_neurons = [Neuron([0.0 for _ in range(64)], 0.0, relu) for _ in range(10)]

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

        print(f"Predicted: {outputs.index(max(outputs))}, Actual: {result}")


    return

if __name__ == "__main__":
    main()
