# Import library
import numpy as np

#Define sigmoid function
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

# Define neural network class
class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation_value = sigmoid
            self.activation_value_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation_value = tanh
            self.activation_value_prime = tanh_prime

# Define neural network layers
        self.weights = []
        for i in range(1, len(layers) - 1):
            r = 4*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        r = 4*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)
        
# Add bias unit to the input layer
def fit(self, X, y, learning_rate=0.4, epochs=300000):        
    ones = np.atleast_2d(np.ones(X.shape[0]))
    X = np.concatenate((ones.T, X), axis=1)
         
    for k in range(epochs):
        i = np.random.randint(X.shape[0])
        a = [X[i]]

        for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.activation_value(dot_value)
                a.append(activation)
            
        error = y[i] - a[-1]
        deltas = [error * self.activation_value_prime(a[-1])]

# Perform computation in the hidden layer
for l in range(len(a) - 2, 0, -1): 
   deltas.append(deltas[1].dot(self.weights[l].T)*self.activation_value_prime(a[l])
deltas.reverse()

# Update weight
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
            if k % 30000 == 0: print ('epochs:', k)
    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)))     
        for l in range(0, len(self.weights)):
            a = self.activation_value(np.dot(a, self.weights[l]))
        return a

# Test model 
if __name__ == '__main__':

    nn = NeuralNetwork([3,3,1])
    X = np.array([[1,0, 0],
                  [0,1, 1],
                  [0,0, 0],
                  [1,1,1],
                  [0,1, 1]])
    y = np.array([1, 0, 0, 1,0])
    nn.fit(X, y)
    for e in X:
        print(e,nn.predict(e))

