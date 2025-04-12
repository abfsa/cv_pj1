# optimizer.py
# optimizer and lr_scheduler for training
import numpy as np

class SGD:
    def __init__(self, params, lr=0.01, weight_decay=0.0):
        self.lr = lr        
        self.weight_decay = weight_decay
        self.params = params

    def step(self):
        for param in self.params:
            layer = param['layer']
            layer.weights -= self.lr * (layer.grad_weights + self.weight_decay * layer.weights)
            layer.bias -= self.lr * layer.grad_bias


class LRScheduler:
    def __init__(self, optimizer, step_size, gamma=0.1):
        """
        参数:
            optimizer: 优化器对象
            step_size: 每隔多少个 epoch 更新一次学习率
            gamma: 每次更新时，学习率乘以的因子
        """
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        """
        调用一次，表示完成了一个epoch。
        到达 step_size 的整数倍时，更新学习率。
        """
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            print(f"Scheduler: Reducing learning rate to {self.optimizer.lr:.6f}")


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # 初始化一阶矩（动量）和二阶矩（方差）
        self.m = {}
        self.v = {}
        self.t = 0  # 记录步数

        for param in self.params:
            layer = param['layer']
            self.m[layer] = {
                'weights': np.zeros_like(layer.weights),
                'bias': np.zeros_like(layer.bias)
            }
            self.v[layer] = {
                'weights': np.zeros_like(layer.weights),
                'bias': np.zeros_like(layer.bias)
            }

    def step(self):
        self.t += 1
        for param in self.params:
            layer = param['layer']

            # 梯度
            grad_w = layer.grad_weights + self.weight_decay * layer.weights
            grad_b = layer.grad_bias + self.weight_decay * layer.bias

            # 更新一阶矩
            self.m[layer]['weights'] = self.beta1 * self.m[layer]['weights'] + (1 - self.beta1) * grad_w
            self.m[layer]['bias'] = self.beta1 * self.m[layer]['bias'] + (1 - self.beta1) * grad_b

            # 更新二阶矩
            self.v[layer]['weights'] = self.beta2 * self.v[layer]['weights'] + (1 - self.beta2) * (grad_w ** 2)
            self.v[layer]['bias'] = self.beta2 * self.v[layer]['bias'] + (1 - self.beta2) * (grad_b ** 2)

            # 偏差修正
            m_hat_w = self.m[layer]['weights'] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m[layer]['bias'] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v[layer]['weights'] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v[layer]['bias'] / (1 - self.beta2 ** self.t)

            # 更新参数
            layer.weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
