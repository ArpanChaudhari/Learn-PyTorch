# PyTorch Day 1 Notes: Fundamentals (Part 1)

## PyTorch Overview

### Introduction & Journey
* **What is PyTorch?** PyTorch is an open-source machine learning library developed primarily by Facebook's AI Research lab (FAIR). It is widely used for applications such as computer vision and natural language processing.
* **The Journey:** PyTorch quickly became the favorite framework for researchers because of its "Pythonic" nature. While earlier frameworks like TensorFlow 1.0 used static computation graphs (where you had to define the entire architecture before running data through it), PyTorch introduced dynamic computation graphs.

### Core Features in PyTorch
1. **Dynamic Computation Graphs (Define-by-Run):** You can change how the network behaves on the fly, which makes debugging incredibly easy compared to static graphs.
2. **GPU Acceleration:** PyTorch provides native support for fast hardware acceleration using NVIDIA GPUs.
3. **Pythonic:** It feels like standard Python. If you know numpy, PyTorch is very easy to pick up.

### PyTorch vs TensorFlow
* **PyTorch:** Preferred by the research community, highly intuitive, dynamic graphs, great for prototyping. (Now widely adopted in industry too).
* **TensorFlow:** Backed by Google, historically better for production deployment (TF Serving) and edge devices (TF Lite). However, PyTorch has caught up with tools like TorchServe.

### Core Modules in PyTorch
* `torch`: The main namespace containing tensor operations (like numpy).
* `torch.nn`: Contains the building blocks for neural networks (layers, loss functions).
* `torch.optim`: Optimization algorithms like SGD, Adam, AdamW.
* `torch.utils.data`: Utilities for data loading and batching (`DataLoader`, `Dataset`).
