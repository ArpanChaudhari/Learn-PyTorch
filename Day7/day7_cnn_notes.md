# PyTorch Day 7 Notes: Convolutional Neural Networks (CNNs)

Artificial Neural Networks (ANNs) struggle with images because they require flattening the 2D image into a 1D vector. This destroys the spatial relationships between pixels (e.g., knowing that a pixel is physically above another pixel). 

Convolutional Neural Networks (CNNs) solve this by preserving the 2D structure and sliding filters over the image to detect features like edges, curves, and eventually complex objects.

## 1. The Core CNN Layers in PyTorch

### `nn.Conv2d` (Convolutional Layer)
This layer applies a set of learnable filters to the input image.
```python
nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
```
* **`in_channels`**: 1 for grayscale images (like MNIST), 3 for colored images (RGB).
* **`out_channels`**: The number of filters you want to use. Each filter learns to detect a different feature. This becomes the `in_channels` for the next layer.
* **`kernel_size`**: The size of the sliding window (e.g., 3 means a $3 \times 3$ matrix).
* **`stride`**: How many pixels the window shifts at a time.
* **`padding`**: Adding zero-pixels around the edge of the image so the filter can properly scan the borders. `padding=1` with `kernel_size=3` keeps the image size exactly the same!

### `nn.MaxPool2d` (Pooling Layer)
Pooling reduces the spatial dimensions (height and width) of the image while keeping the most important information. It saves memory and prevents overfitting.
```python
nn.MaxPool2d(kernel_size=2, stride=2)
```
* This operation cuts the image height and width exactly in half. A $28 \times 28$ image becomes $14 \times 14$.

## 2. Building the CNN Architecture

A standard CNN typically follows this pattern:
`Conv2d` -> `ReLU` -> `MaxPool2d` (repeated) -> `Flatten` -> `Linear` -> `Linear` (Output)

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature Extractor
        self.conv_block = nn.Sequential(
            # Input: (1 channel, 28x28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: (16 channels, 28x28)
            
            nn.MaxPool2d(kernel_size=2),
            # Output: (16 channels, 14x14)
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # Final output: (32 channels, 7x7)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # The input here requires calculating the final flattened size: 32 * 7 * 7
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # 10 classes
        )
        
    def forward(self, x):
        # Pass through convolutions
        x = self.conv_block(x)
        # Pass through linear layers
        x = self.classifier(x)
        return x
```

## 3. The Flattening Transition (The tricky part)
When moving from the `conv_block` to the `classifier` (the linear layers), you must flatten the tensor. 
To determine the input features for your first `nn.Linear` layer, you multiply:
**`(final_out_channels) * (final_height) * (final_width)`**

If you miscalculate this number, PyTorch will throw a `RuntimeError: mat1 and mat2 shapes cannot be multiplied`. A great debugging trick is to print the shape of `x` inside the `forward` function right before passing it to the linear layer!
