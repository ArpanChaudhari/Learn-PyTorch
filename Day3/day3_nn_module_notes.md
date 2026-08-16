# PyTorch Day 3 Notes: `nn.Module` & The Built-in Training Pipeline

After manually coding the training loop and calculating gradients, the next step is using PyTorch's built-in tools which abstract away the complex math while keeping the code readable.

## 1. Subclassing `nn.Module`
`torch.nn.Module` is the base class for all neural network models in PyTorch.
* You must inherit from it: `class MyModel(nn.Module):`
* You define your layers in the `__init__` method.
* You define how data flows through the layers in the `forward` method.

```python
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        # PyTorch automatically initializes and tracks these weights!
        self.linear = nn.Linear(in_features=input_features, out_features=1)
        
    def forward(self, x):
        # We don't need to manually do torch.matmul(x, w) + b anymore
        return self.linear(x)
```

## 2. Built-in Layers & Activations
* **`nn.Linear(in, out)`:** Performs a linear transformation (y = xW^T + b). It automatically initializes weights and biases and tracks their gradients.
* **Activations:** PyTorch has built-in activations like `torch.sigmoid()` (functional API) or `nn.Sigmoid()` (object-oriented layer).

## 3. Built-in Loss Functions
Instead of writing the cross-entropy formula by hand, PyTorch provides optimized implementations:
* **`nn.MSELoss()`:** Mean Squared Error (for regression).
* **`nn.BCELoss()`:** Binary Cross Entropy (requires the output to be passed through a sigmoid first).
* **`nn.BCEWithLogitsLoss()`:** Combines a Sigmoid layer and the BCELoss in one single class. This is numerically more stable than using a plain Sigmoid followed by a BCELoss.

```python
criterion = nn.BCELoss()
loss = criterion(y_pred, y_true)
```

## 4. Built-in Optimizers
Instead of manually updating the weights (`weights -= lr * grad`), PyTorch optimizers handle the parameter updates automatically.
* Popular optimizers: `torch.optim.SGD`, `torch.optim.Adam`, `torch.optim.AdamW`.
* You pass the model's parameters (`model.parameters()`) to the optimizer so it knows exactly what to update.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

## 5. The Clean PyTorch Training Loop
With `nn.Module`, Loss Functions, and Optimizers, the 5-step loop becomes much cleaner:

```python
for epoch in range(epochs):
    # 1. Forward Pass
    y_pred = model(X_train) 
    
    # 2. Calculate Loss
    loss = criterion(y_pred, y_train)
    
    # 3. Zero the Gradients
    optimizer.zero_grad()
    
    # 4. Backward Pass (Calculate gradients)
    loss.backward()
    
    # 5. Step the Optimizer (Update weights)
    optimizer.step()
```

This structure is universal. Whether you are training a simple logistic regression model or a billion-parameter Transformer, this exact 5-step loop remains exactly the same!
