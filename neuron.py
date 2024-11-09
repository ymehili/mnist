import math
from typing import List


class Neuron:
    def __init__(self, weights: List[float], bias: float, activation_function):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs: List[float]) -> float:
        # Calculate the weighted sum of inputs + bias
        z = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.activation_function(z)
    


def relu(self, z: float) -> float:
    # Using ReLU as the activation function for hidden layers
    return max(0, z)

def sigmoid(self, z: float) -> float:
    # Using sigmoid as the activation function for output layer
    return 1 / (1 + math.exp(-z))

def softmax(self, z: List[float]) -> List[float]:
    # Using softmax as the activation function for output layer
    exp_values = [math.exp(i) for i in z]
    total = sum(exp_values)
    return [i / total for i in exp_values]
