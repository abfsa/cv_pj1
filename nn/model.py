import numpy as np
import pickle

class Model:
    def __init__(self, layers, optimizer = None, optimizer_params = None , hyperparams = None):
        self.layers = layers
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.hyperparams = hyperparams if hyperparams is not None else {}

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def __call__(self, X):
        return self.forward(X)

    def get_params(self):
        "获取模型的参数，用于优化器"
        params = []
        for layer in self.layers:
            if hasattr(layer, 'optimizable') and layer.optimizable:
                params.append({'params': [layer.weights, layer.bias], 'layer': layer})
        return params
    
    def train(self):
        "设置模型为训练状态，激活dropout"
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
    
    def eval(self):
        "设置模型为评估状态，关闭dropout"
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'optimizable') and layer.optimizable:
                layer.grad_weights = np.zeros_like(layer.weights)
                layer.grad_bias = np.zeros_like(layer.bias)
                layer.grad_input = np.zeros_like(layer.input)

    def save(self, filepath):
        """save model to filepath， 
        save_dict = {'params' : 模型参数, 
        'optimizer_params' : 优化器参数,
        'hyperparams' : 超参数} """
        save_dict = {
            'params' : {},
            'optimizer_params' : self.optimizer_params,
            'hyperparams' : self.hyperparams
        }

        for index, layer in enumerate(self.layers):
            if hasattr(layer, 'optimizable') and layer.optimizable:
                layer_name = layer.__class__.__name__ + '_' + str(index)
                save_dict['params'][layer_name] = {'weights' : layer.weights, 'bias' : layer.bias}
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)

    def load(self, filepath):
        "load model from filepath"
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        for index, layer in enumerate(self.layers):
            if hasattr(layer, 'optimizable') and layer.optimizable:
                layer_name = layer.__class__.__name__ + '_' + str(index)
                if layer_name in save_dict['params']:
                    param = save_dict['params'][layer_name]
                    layer.weights = param['weights']
                    layer.bias = param['bias']
        self.optimizer_params = save_dict['optimizer_params']
        self.hyperparams = save_dict['hyperparams']