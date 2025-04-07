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
        super.__init__()
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = np.maximum(0, X)
        return self.output
    
    def backward(self, output_gradient):
        return (self.input > 0) * output_gradient


class Softmax(Layer):
    def __init__(self):
        super.__init__()
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
        return self.output

    def backward():
        pass



class CrossEntropy(Layer):
    def __init__(self):
        super.__init__()
        self.input = None
        self.output = None

    def forward(self, X, y):
        self.input = X
        self.output = -np.sum(y * np.log(X)) / X.shape[0]
        return self.output

    def backward():
        pass


class Linear(Layer):
    def __init__(self, input_size, output_size):
        super.__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))
        self.input = None
        self.output = None

    def forward(self, X):