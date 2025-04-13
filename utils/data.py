import os
import numpy as np
import pickle

class Dataset:
    def __init__(self, data, labels, transform = None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
        if self.transform:
            data = self.transform(data)
        return data, label


class CIFAR10Dataset(Dataset):
    """加载CIFAR10数据集"""
    def __init__(self, root, train=True, transform = None):
        self.root = root
        self.train = train
        data, lables = self.load_data()
        super().__init__(data, lables, transform)
    
    def load_data(self):
        if self.train:
            files = [f'data_batch_{i}' for i in range(1,6)]
        else:
            files = ['test_batch']

        data_list = []
        label_list = []

        for file in files:
            file_path = os.path.join(self.root, file)
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
                data = data_dict[b'data']
                labels = data_dict[b'labels']

                data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                data_list.append(data)
                label_list.append(labels)

        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(label_list)
        return data, labels
    
def train_val_split(dataset, val_ratio=0.2, shuffle=True):
    """将数据集按比例划分为训练集和验证集"""
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
    
    split = int(len(dataset) * (1 - val_ratio))
    train_indices, val_indices = indices[:split], indices[split:]

    train_data = dataset.data[train_indices]
    train_labels = dataset.labels[train_indices]
    val_data = dataset.data[val_indices]
    val_labels = dataset.labels[val_indices]

    train_set = Dataset(train_data, train_labels, transform=dataset.transform)
    val_set = Dataset(val_data, val_labels, transform=dataset.transform)

    return train_set, val_set

def dataloader_generator(dataset, batch_size=32, shuffle=True):
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(indices), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        
        batch_data = []
        batch_labels = []

        for idx in batch_indices:
            data, label = dataset[idx]
            batch_data.append(data)
            batch_labels.append(label)

        yield np.array(batch_data), np.array(batch_labels)


def to_float32(x):
    return x.astype(np.float32)

def normalize(x):
    # 归一化到 [0,1]
    return x / 255.0

def standardize(x):
    # 标准化 (cifar-10 训练集的均值和标准差)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    return (x / 255.0 - mean) / std

def compose(*transforms):
    def composed_transform(x):
        for t in transforms:
            x = t(x)
        return x
    return composed_transform

def random_flip(x):
    """随机水平翻转图像"""
    if np.random.rand() > 0.5:
        return np.flip(x, axis=1)  # 水平翻转
    return x

def random_crop(x, padding=4):
    """随机裁剪图像，先在边缘padding再裁剪"""
    H, W, C = x.shape
    x_padded = np.pad(x, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    top = np.random.randint(0, 2 * padding)
    left = np.random.randint(0, 2 * padding)
    return x_padded[top:top+H, left:left+W, :]

def random_noise(x, noise_level=0.05):
    """加随机高斯噪声"""
    noise = np.random.randn(*x.shape) * noise_level
    x_noisy = x + noise
    return np.clip(x_noisy, 0.0, 1.0)  # 保持在[0,1]

def random_brightness(x, brightness_range=0.2):
    """随机调整亮度"""
    factor = 1.0 + np.random.uniform(-brightness_range, brightness_range)
    x = x * factor
    return np.clip(x, 0.0, 1.0)