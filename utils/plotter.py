import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class Plotter:
    def __init__(self):
        self.train_acc = []
        self.val_acc = []
        self.test_acc = []
        self.train_loss = []
        self.val_loss = []
        
        # 初始化画布
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 5))
        # plt.ion()  # 打开交互模式，实时更新
        
    def update(self, train_acc=None, val_acc=None, test_acc=None, train_loss=None, val_loss=None):
        if train_acc is not None:
            self.train_acc.append(train_acc)
        if val_acc is not None:
            self.val_acc.append(val_acc)
        if test_acc is not None:
            self.test_acc.append(test_acc)
        if train_loss is not None:
            self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        
        self.plot()
        
    def plot(self):
        self.axs[0].clear()
        self.axs[1].clear()
        
        # Accuracy 曲线
        self.axs[0].plot(self.train_acc, label='Train Acc')
        self.axs[0].plot(self.val_acc, label='Val Acc')
        if len(self.test_acc) > 0:
            self.axs[0].plot(self.test_acc, label='Test Acc')
        self.axs[0].set_title('Accuracy')
        self.axs[0].set_xlabel('Epoch')
        self.axs[0].set_ylabel('Accuracy')
        self.axs[0].legend()
        self.axs[0].grid(True)
        
        # Loss 曲线
        self.axs[1].plot(self.train_loss, label='Train Loss')
        if len(self.val_loss) > 0:
            self.axs[1].plot(self.val_loss, label='Val Loss')
        self.axs[1].set_title('Loss')
        self.axs[1].set_xlabel('Epoch')
        self.axs[1].set_ylabel('Loss')
        self.axs[1].legend()
        self.axs[1].grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # 短暂停一下才能实时刷新
        
    def save(self, path='training_curve.png'):
        self.fig.savefig(path)
        print(f"Plot saved to {path}")
        
    def close(self):
        plt.close()
