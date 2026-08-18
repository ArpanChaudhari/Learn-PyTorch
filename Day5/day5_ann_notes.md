# PyTorch Day 5 Notes: Building an ANN (Multi-Class Classification)

Today's focus was bridging the gap between basic linear regression and Deep Learning by building a Multi-Layer Perceptron (ANN) to classify clothing items using the Fashion-MNIST dataset.

## 1. Multi-Class Classification Architecture
Unlike binary classification which outputs a single probability (using Sigmoid), Multi-Class classification outputs a probability distribution across multiple classes (using Softmax).

When building the `nn.Module` for Fashion-MNIST (10 classes of 28x28 images):
* **Input Layer:** Must match the flattened image size ($28 \times 28 = 784$ features).
* **Hidden Layers:** Allow the network to learn non-linear representations. We use `nn.Linear()` stacked together.
* **Output Layer:** Must output exactly $10$ features (one raw score/logit for each class).

```python
import torch.nn as nn

class FashionANN(nn.Module):
    def __init__(self):
        super().__init__()
        # Flattening 2D images (28x28) into 1D vectors (784)
        self.flatten = nn.Flatten()
        
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),           # Non-linear activation
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)    # 10 output classes
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits
```

## 2. Activation Functions (`nn.ReLU`)
If we only stack `nn.Linear` layers, the entire network simply collapses back down to a single linear transformation. By inserting `nn.ReLU()` (Rectified Linear Unit) between the hidden layers, the network gains the mathematical capability to model highly complex, non-linear boundaries.

## 3. Loss Function: `nn.CrossEntropyLoss`
For multi-class problems, we use `nn.CrossEntropyLoss()`.
* **Important PyTorch Detail:** This function expects **raw, unnormalized logits** directly from the final `nn.Linear` layer. 
* Do **NOT** put a `nn.Softmax()` at the end of your model when using `nn.CrossEntropyLoss()`. PyTorch applies `LogSoftmax` internally for numerical stability.

## 4. Evaluation (Accuracy)
During evaluation, we extract the predicted class by finding the index of the highest probability using `torch.argmax()`.

```python
with torch.no_grad():
    y_pred_logits = model(X_test)
    # Get the index of the max logit (the predicted class label)
    predicted_classes = torch.argmax(y_pred_logits, dim=1)
    
    accuracy = (predicted_classes == y_test).float().mean()
```

## 5. Summary of the Pipeline
1. Load dataset (Fashion-MNIST).
2. Wrap in `Dataset` and `DataLoader` for batching.
3. Define the ANN with `nn.Sequential` and `nn.ReLU`.
4. Choose `nn.CrossEntropyLoss()` and `torch.optim.Adam()`.
5. Run the 5-step training loop across multiple epochs.
6. Track Training/Validation Loss and Accuracy.
