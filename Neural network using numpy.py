import numpy as np

# Define the XOR gate dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define hyperparameters
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 10000

# Initialize weights and biases
w1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
w2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    layer1_input = np.dot(X, w1) + b1
    layer1_output = sigmoid(layer1_input)
    layer2_input = np.dot(layer1_output, w2) + b2
    layer2_output = sigmoid(layer2_input)

    # Calculate the loss
    loss = np.mean((layer2_output - y) ** 2)

    # Backpropagation
    d_loss = 2 * (layer2_output - y)
    d_layer2_input = d_loss * sigmoid_derivative(layer2_output)
    
    d_layer1_output = d_layer2_input.dot(w2.T)
    d_layer1_input = d_layer1_output * sigmoid_derivative(layer1_output)

    # Update weights and biases
    w2 -= layer1_output.T.dot(d_layer2_input) * learning_rate
    b2 -= np.sum(d_layer2_input, axis=0, keepdims=True) * learning_rate
    w1 -= X.T.dot(d_layer1_input) * learning_rate
    b1 -= np.sum(d_layer1_input, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}: Loss = {loss:.4f}')

# Testing the trained neural network
layer1_input = np.dot(X, w1) + b1
layer1_output = sigmoid(layer1_input)
layer2_input = np.dot(layer1_output, w2) + b2
layer2_output = sigmoid(layer2_input)

predicted_labels = (layer2_output > 0.5).astype(int)
print("Predicted labels:")
print(predicted_labels)
