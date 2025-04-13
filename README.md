# cv_pj1
**dnn for cifar-10 classify**

## Install requirements

```bash
pip install numpy tqdm
```

## How to train?

1. Prepare the CIFAR-10 dataset files under `dataset/cifar-10-batches-py/` (binary format).
2. Run the training script:

```bash
python train.py
```

- Model checkpoints will be saved under `saved_model/`.
- Training curves (optional if plotting is enabled) saved under `figs/`.
- Hyperparameter search results saved in `saved_model/search_results.json`

## How to test?

Examples for testing and visualizing are prepared in the test.ipynb.

## Directory Structure

- **nn/**

  - **layers.py**
     Basic building blocks (layers) are defined here for the neural network, including an abstract `Layer`, `ReLu`, `Sigmoid`, `Softmax`, `CrossEntropy`, `Linear`, and `Dropout`. Each layer supports forward and backward propagation, and `Linear` layers are marked as optimizable for training.

  - **optimizer.py**
     Optimizers and learning rate schedulers for training networks are implemented here, including `SGD`, `Adam`, and `LRScheduler`. Optimizers update model parameters based on gradients computed during backpropagation.

  - **model.py**
     The `Model` class is defined here. It provides the complete training and inference pipeline. One can initialize a model using code such as:

    ```python
    model = Model([Linear(input, hidden), Sigmoid(), Linear(hidden, output), Softmax()])
    ```

    Then train, save, and load using the implemented methods of the class.

- **utils/**

  - **data.py**
     This file provides data handling utilities, particularly for loading and preparing the CIFAR-10 dataset. Some useful functions are provided for train-validation splitting, data transformation, and dataloader generation. Transformations include standardizing, normalizing, and some data augmentation methods such as flipping, cropping, etc.
  - **metric.py**
     Metric computation utilities for evaluating model performance are implemented in this file. One can use `evaluate_accuracy_and_loss` to evaluate the accuracy and loss of a given model and a specified dataset.
  - **plotter.py**
     The `Plotter` class for visualizing training progress is defined here.
