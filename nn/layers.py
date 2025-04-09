import numpy as np

class Layer:
    def __init__(self):
        self.optimizable = False

    def forward(self, input):
        raise NotImplementedError()
    
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError()

    def __call__(self, X):
        return self.forward(X)
    
class ReLu(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = np.maximum(0, X)
        return self.output
    
    def backward(self, grad_output):
        return (self.input > 0) * grad_output


class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        return grad_output * (self.output * (1 - self.output))
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward(self, grad_output):
        return grad_output * (self.output * (1 - self.output))


class CrossEntropy(Layer):
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None

    def forward(self, X, y):
        self.input = X
        self.label = y
        self.batchsize = X.shape[0]
        self.output = -np.sum(y * np.log(X)) / self.batchsize
        return self.output

    def backward(self):
        return (self.input - self.label) / self.batchsize

class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        self.input = None
        self.output = None
        self.optimizable = True
        self.grad_weights = None
        self.grad_bias = None
        self.batchsize = None

    def forward(self, X):
        self.input = X
        self.batchsize = X.shape[0]
        self.output = np.dot(X, self.weights) + self.bias
        return self.output


    def backward(self, grad_output):
        grad_input = np.matmul(grad_output, self.weights.T)
        self.grad_weights = np.matmul(self.input.T, grad_output) / self.batchsize
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True) / self.batchsize
        return grad_input
    
    