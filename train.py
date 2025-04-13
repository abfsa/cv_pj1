import numpy as np
from nn import SGD, LRScheduler, Adam
from nn import Model
from nn import CrossEntropy, Softmax, ReLu, Linear, Dropout, Sigmoid
from utils import *
from tqdm import tqdm
import os
import itertools
import json

# ====== 目录设置 ======
save_dir = r'saved_model'
fig_dir = r'figs'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
result_file = os.path.join(save_dir, 'search_results_deeper.json')
file_path = r'dataset\cifar-10-batches-py'

# ====== 超参数网格 ======
hidden_sizes = [(512, 128)]
lrs = [0.001]
weight_decays = [0.001]
schedulers = [(5, 0.8)]
dropouts = [0.0]
batch_size = 64
num_epochs = 20
patience = 5

# ====== 加载数据集 ======
full_train_set = CIFAR10Dataset(file_path, train=True, transform=standardize)
train_set, val_set = train_val_split(full_train_set)

# ====== 超参数搜索 ======

search_results = []

for (hidden_size_1, hidden_size_2), lr, weight_decay, (step_size, gamma), dropout in itertools.product(hidden_sizes, lrs, weight_decays, schedulers, dropouts):
    print(f"\nTraining with hidden_size={hidden_size_1} and {hidden_size_2}, lr={lr}, weight_decay={weight_decay}, step_size={step_size}, gamma={gamma}, dropout_rate={dropout}")
    layers = [Linear(3072, hidden_size_1), ReLu(), Linear(hidden_size_1, hidden_size_2), ReLu() , Linear(hidden_size_2, 10), Softmax()]
    model = Model(layers)
    optimizer = Adam(model.get_params(), lr=lr, weight_decay=weight_decay)
    scheduler = LRScheduler(optimizer, step_size=step_size, gamma=gamma)
    loss = CrossEntropy(10)
    batchsize = 64
    plotter = Plotter()
    patience = 5
    best_val_loss = float('inf')
    wait = 0

    model_tag = f"adam_h{hidden_size_1}_{hidden_size_2}_lr{lr}_wd{weight_decay}_ss{step_size}_g{gamma}_d{dropout}_deeper".replace('.', '_')
    model_save_path = os.path.join(save_dir, model_tag)
    fig_save_path = os.path.join(fig_dir, f"{model_tag}.png")
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in tqdm(range(20)):
        model.train()
        for X, y in dataloader_generator(train_set, batch_size=batchsize, shuffle=True):
            batch_X = X.reshape(X.shape[0], -1)
            y_pred = model(batch_X)
            l = loss.forward(y_pred, y)
            grad = loss.backward()
            model.backward(grad)
            optimizer.step()
        train_acc, train_loss = evaluate_accuracy_and_loss(model, dataloader_generator(train_set, batch_size=128, shuffle=False), loss)
        val_acc, val_loss = evaluate_accuracy_and_loss(model, dataloader_generator(val_set, batch_size=128, shuffle=False), loss)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(os.path.join(model_save_path,'model.pkl'))
            print(f"Validation loss improved. Saving model to {model_save_path}.")
            wait = 0
        else:
            wait += 1
            print(f"No improvement in validation loss. Wait {wait}/{patience}")
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        plotter.update(train_acc= train_acc, train_loss=train_loss, val_loss=val_loss, val_acc=val_acc)
        scheduler.step()

    plotter.save(fig_save_path)
    plotter.close()

    search_results.append({
        "dropout": dropout,
        "hidden_size": hidden_sizes,
        "lr": lr,
        "weight_decay": weight_decay,
        "step_size": step_size,
        "gamma": gamma,
        "best_val_loss": best_val_loss
    })

os.makedirs(save_dir, exist_ok=True)
with open(result_file, 'w') as f:
    json.dump(search_results, f, indent=4)

print("\n超参数搜索结束，结果已保存到：", result_file)