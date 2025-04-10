import numpy as np

class Layer:
    def __init__(self):
        self.optimizable = False

    def forward(self, input):
        raise NotImplementedError()
    
    def backward(self, grad_output):
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
        return grad_output # softmax 和 cross entropy 一起使用， 梯度在cross entropy里面计算，这里直接回传
    
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
    def __init__(self, classifier = 'softmax'):
        super().__init__()
        self.input = None
        self.output = None
        self.classifier = classifier

    def forward(self, X, y):
        self.input = X
        self.label = y
        self.batchsize = X.shape[0]
        self.output = -np.sum(y * np.log(X)) / self.batchsize
        return self.output

    def backward(self, grad_output = None):
        if self.classifier == 'softmax':
            return (self.input - self.label) / self.batchsize
        else:
            raise NotImplementedError()

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
    
class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.training = True


    def forward(self, X):
        if self.training:
            self.mask = (np.random.rand(*X.shape) > self.p).astype(float)
            return X *self.mask /(1.0 - self.p)
        else:
            return X
    
    def backward(self, grad_output):
        if self.training:
            return grad_output * self.mask / (1.0 - self.p)
        else:
            return grad_output