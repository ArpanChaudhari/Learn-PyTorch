# PyTorch Day 9 Notes: Fine-Tuning MobileNetV3-Small

Today we shifted focus to a new dataset (**10 Monkey Species**) and applied **Fine-Tuning** using a lightweight, highly efficient pre-trained model: **MobileNetV3-Small**.

## 1. Custom Dataset & Data Loading
When dealing with images stored in categorical folders (e.g., `n0`, `n1`, `n2` for different monkey species), we must dynamically load the file paths and map them to integer labels.

```python
# Function to traverse folders and build a Pandas DataFrame of (filepath, label)
def build_Xy_dataframe(directory):
    X_filepaths, y_labels = [], []
    class_folders = sorted(os.listdir(directory))
    
    for label_number, folder_name in enumerate(class_folders):
        folder_path = os.path.join(directory, folder_name)
        for image_name in os.listdir(folder_path):
            X_filepaths.append(os.path.join(folder_path, image_name))
            y_labels.append(label_number)
            
    return pd.DataFrame({"X_filepath": X_filepaths, "y_label": y_labels})
```
Inside the `CustomDataset`'s `__getitem__`, we use `Image.open(img_path).convert("RGB")` to safely load the images before applying the `transforms.Compose` pipeline (Resize to 256, CenterCrop to 224, ToTensor, Normalize).

## 2. Why MobileNetV3-Small?
Unlike VGG16 (which is over 500MB and computationally heavy), MobileNet architectures use **Depthwise Separable Convolutions**. This drastically reduces the number of parameters, making it possible to run deep learning models in real-time on edge devices (like mobile phones) without sacrificing much accuracy.

## 3. Fine-Tuning the Architecture
We load the pre-trained MobileNetV3-Small and modify its classifier.

```python
import torchvision.models as models

# 1. Load the pre-trained MobileNetV3-Small
mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

# 2. Freeze the feature extraction layers
for param in mobilenet.features.parameters():
    param.requires_grad = False

# 3. Replace the final classification head
# MobileNetV3-Small's classifier is an nn.Sequential block. 
# The input to the last Linear layer is 1024 features.
import torch.nn as nn

mobilenet.classifier[3] = nn.Linear(in_features=1024, out_features=10) # 10 Monkey Species
```

## 4. Fine-Tuning vs Transfer Learning
While standard Transfer Learning only trains the final newly-added classification head, **Fine-Tuning** takes it a step further. 
After the new head converges, you can *unfreeze* the last few convolutional blocks of the pre-trained model (`requires_grad = True`) and train them with a very low learning rate (e.g., `1e-5`). This allows the edge-detectors to slightly adapt their weights specifically to monkey fur and facial features, pushing the accuracy even higher!
