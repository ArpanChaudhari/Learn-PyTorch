# PyTorch Day 8 Notes: Transfer Learning

Training deep Convolutional Neural Networks (CNNs) from scratch requires massive datasets (millions of images) and immense computational power. 

**Transfer Learning** solves this by taking a model that has already been trained on a massive dataset (like ImageNet, which has 1,000 classes) and repurposing its learned feature extractors (edge detectors, shape detectors) for a new, custom dataset.

## 1. Loading a Pre-Trained Model
PyTorch provides state-of-the-art models via `torchvision.models`. To use transfer learning, you must load the model with its pre-trained weights.

```python
import torchvision.models as models

# Load a ResNet18 model trained on ImageNet
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
```

## 2. Freezing the Base Layers (Feature Extractor)
We do not want to destroy the weights the model spent weeks learning. Therefore, we "freeze" the early layers of the network by turning off gradient tracking.

```python
for param in model.parameters():
    param.requires_grad = False
```
Because `requires_grad` is `False`, the optimizer will completely ignore these layers during the `.backward()` pass, saving massive amounts of memory and compute time.

## 3. Modifying the Classification Head
A pre-trained model like ResNet18 outputs 1,000 classes by default. If your custom dataset only has 2 classes (e.g., Cats vs Dogs), you must replace the final fully-connected (`fc`) layer.

```python
import torch.nn as nn

# Find the number of input features going into the final layer
num_features = model.fc.in_features

# Replace the final layer with a NEW layer. 
# New layers have requires_grad=True by default!
model.fc = nn.Linear(num_features, 2) 
```

## 4. Training the Model
Because the early layers are frozen, we only need to pass the parameters of the *new* classification head to the optimizer.

```python
import torch.optim as optim

# Only optimize the parameters of the newly added 'fc' layer
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

When you run your standard 5-step PyTorch training loop, only the final layer will update. The model acts as a highly advanced feature extractor, pushing your images through the frozen layers, and then the final layer learns how to map those extracted features to your specific classes.

## 5. Fine-Tuning (Advanced)
Once the new classification head has been trained for a few epochs, you can optionally "unfreeze" some of the later convolutional layers (by setting `requires_grad = True`) and train the entire network with a very small learning rate (e.g., `1e-5`) to fine-tune the feature extractors specifically for your dataset.
