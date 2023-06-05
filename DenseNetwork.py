import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)
        self.activation = activation
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases
        self.activation_output = self.activation_func(self.output)

        return self.activation_output
    
    def backward(self, grad_output, learning_rate):
        grad_output *= self.activation_grad()
        grad_weights = np.dot(self.inputs.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)
        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input
    
    def activation_func(self, z):
        if self.activation == 'sigmoid':
            return 1 / (1+np.exp(-z))
        if self.activation == 'relu':
            return np.maximum(z, 0)
        return z
    
    def activation_grad(self):
        if self.activation == 'sigmoid':
            return self.activation_output * (1-self.activation_output)
        if self.activation == 'relu':
            return np.where(self.activation_output>0, 1, 0)
        return 1
    
    
class DenseNetwork:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=10000, n_features=20, noise=0.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dense_net = DenseNetwork()
dense_net.add_layer(DenseLayer(20, 100, 'relu'))
dense_net.add_layer(DenseLayer(100, 1))

learning_rate = 0.001
num_epochs = 1000
for epoch in range(num_epochs):
    
    y_pred = dense_net.forward(X_train_scaled)
    
    loss = np.mean((y_pred - y_train) ** 2)
    if epoch %100 == 0:
        print(f'epoch {epoch}:{loss}', end='\r')
    grad_output = (y_pred - y_train) / len(X_train_scaled)
    dense_net.backward(grad_output, learning_rate)

y_pred_dense = dense_net.forward(X_test_scaled)

print("Mean Squared Error (DenseNetwork implemented from scratch):", mean_squared_error(y_test, y_pred_dense))

