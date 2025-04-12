import numpy as np

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_pred, y):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return (float(np.sum(y_pred == y)))

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    net.eval()
    metric = Accumulator(2)
    for X, y in data_iter:
        batch_X = X.reshape(X.shape[0], -1)
        metric.add(accuracy(net(batch_X), y), len(y))
    return metric[0] / metric[1]

def evaluate_accuracy_and_loss(model, data_iter, loss_fn):
    """同时计算模型在指定数据集上的accuracy和loss"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for X, y in data_iter:
        batch_X = X.reshape(X.shape[0], -1)
        y_pred = model(batch_X)
        
        # Loss
        l = loss_fn.forward(y_pred, y)
        total_loss += l * len(y)
        
        # Accuracy
        total_correct += accuracy(y_pred, y)
        
        total_samples += len(y)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_acc, avg_loss
