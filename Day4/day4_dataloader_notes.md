# PyTorch Day 4 Notes: Dataset & DataLoader

Real-world datasets are too large to fit entirely into GPU memory. PyTorch solves this by batching data using `Dataset` and `DataLoader`.

## 1. Custom `Dataset`
The `torch.utils.data.Dataset` class is an abstract class representing a dataset. When creating a custom dataset, you must override two critical methods:
* `__len__(self)`: Returns the total number of samples in the dataset.
* `__getitem__(self, idx)`: Returns a single sample (and its label) at the given index.

```python
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

## 2. `DataLoader` Properties
The `torch.utils.data.DataLoader` wraps an iterable around the Dataset to enable easy access to samples. It takes several important parameters to control how data is fed into the model:

1. **`dataset`**: The Dataset object to load data from.
2. **`batch_size`**: The number of samples to load per batch. (e.g., 32, 64, 128). This dictates how many inputs the model processes before a backward pass and weight update.
3. **`shuffle`**: Set to `True` for the training set so the model doesn't learn the sequence of the data. Set to `False` for the validation/test set.
4. **`num_workers`**: Defines how many subprocesses to use for data loading. `0` means the data will be loaded in the main process. Increasing this speeds up loading for large datasets (like images).
5. **`drop_last`**: If `True`, drops the last incomplete batch if the dataset size is not divisible by the batch size. This prevents shape mismatch errors during training if your model architecture expects a fixed batch size.
6. **`collate_fn`**: A function that merges a list of samples to form a mini-batch of Tensors. PyTorch has a default `collate_fn`, but you can write a custom one (crucial for NLP when padding text sequences of different lengths).
7. **`sampler`**: Defines the strategy to draw samples from the dataset. If specified, `shuffle` must not be specified. Useful for weighted sampling (handling imbalanced datasets).

## 3. Training Pipeline with DataLoader
With the DataLoader, your training loop now has an inner loop that iterates through the batches.

```python
epochs = 10
for epoch in range(epochs):
    # Inner loop over batches!
    for batch_features, batch_labels in dataloader:
        
        # 1. Forward Pass
        y_pred = model(batch_features)
        
        # 2. Calculate Loss
        loss = criterion(y_pred, batch_labels)
        
        # 3. Zero Gradients
        optimizer.zero_grad()
        
        # 4. Backward Pass
        loss.backward()
        
        # 5. Optimizer Step
        optimizer.step()
```
