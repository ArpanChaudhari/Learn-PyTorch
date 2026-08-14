# PyTorch Day 1 Notes: Tensors (Part 2)

A **Tensor** is the fundamental data structure in PyTorch. It is essentially a multi-dimensional array, very similar to a NumPy array, but with the added superpower that it can run on a GPU.

### 1. Creating Tensors
You can create tensors in several ways:
* **Uninitialized/Filled:** `torch.empty(2,3)`, `torch.zeros(2,3)`, `torch.ones(3,3)`, `torch.full((3,3), 5)`
* **Randomized:** `torch.rand(2,3)` (Uniform distribution). *Note: Use `torch.manual_seed(100)` for reproducible random numbers.*
* **From Python Lists:** `torch.tensor([[1, 2, 3], [4, 5, 6]])`
* **Sequences:** `torch.arange(1, 10, 2)` (step-based), `torch.linspace(2, 10, 10)` (equally spaced).
* **Identity Matrix:** `torch.eye(5)`
* **Like another tensor:** `torch.empty_like(x)`, `torch.zeros_like(x)`, `torch.rand_like(x)`

### 2. Tensor Properties
* **Shape:** Access using `.shape`. (e.g., `torch.Size([3, 3])`).
* **Data Type:** Access using `.dtype`. You can specify it during creation (`dtype=torch.float32`) or cast it later using `x.to(torch.float32)`.

### 3. Mathematical Operations
* **Scalar Operations:** Standard math operators apply element-wise (`x + 2`, `x * 3`, `x ** 2`, `torch.abs(c)`, `torch.round(d)`).
* **Element-wise Operations:** `a + b`, `a * b` (multiplies corresponding elements).
* **Reduction Operations:** Operations that reduce dimensions:
  * `torch.sum(e)`, `torch.mean(e)`, `torch.min(e)`, `torch.max(e)`
  * *Crucial concept:* Use `dim=0` to reduce along columns, and `dim=1` to reduce along rows.
  * Statistical: `torch.std(e)`, `torch.var(e)`
  * Positional: `torch.argmax(e)` (returns the index of the maximum value).
* **Matrix Operations:**
  * Dot Product: `torch.dot(v1, v2)` (1D tensors only).
  * Matrix Multiplication: `torch.matmul(f, g)`
  * Matrix Transpose: `torch.transpose(f, 0, 1)`
  * Advanced Math: `torch.det(h)` (Determinant), `torch.inverse(h)`.

### 4. Special Functions (Activation functions)
PyTorch has built-in math functions commonly used in Deep Learning:
* `torch.log(k)`, `torch.exp(k)`, `torch.sqrt(k)`
* **Activations:** `torch.sigmoid(k)`, `torch.softmax(k, dim=0)`, `torch.relu(k)`

### 5. Inplace Operations
In PyTorch, operations that mutate a tensor in-place end with an underscore (`_`).
* Standard: `m + n` (Creates a new tensor in memory).
* Inplace: `m.add_(n)` (Adds `n` to `m`, modifying `m` directly and saving memory).
* `m.relu_()`

### 6. Copying Tensors
* **Warning:** Doing `b = a` does NOT create a copy. Both variables point to the same memory address (`id(a) == id(b)`). Modifying `a` will modify `b`.
* **Correct way:** Use `b = a.clone()` to create an independent copy in memory.

### 7. Tensor Operations on GPU
Tensors run on the CPU by default. To unlock massive speedups (as seen in your notebook: ~129x faster matrix multiplication!):
1. Check availability: `torch.cuda.is_available()`
2. Define device: `device = torch.device('cuda')`
3. Move tensor: `b = a.to(device)` or create it directly `torch.rand((2,3), device=device)`

### 8. Reshaping Tensors
Often, neural network layers expect specific shapes.
* **`reshape()`:** Changes dimensions (e.g., `a.reshape(2,2,2,2)`). The total number of elements must remain the same.
* **`flatten()`:** Converts a multi-dimensional tensor into a 1D tensor.
* **`permute()`:** Swaps dimensions (useful for images). E.g., `b.permute(2, 1, 0)`.
* **`unsqueeze(dim)`:** Adds a dimension of size 1. (e.g., turning an image `[226, 226, 3]` into a batch `[1, 226, 226, 3]`).
* **`squeeze()`:** Removes dimensions of size 1.

### 9. NumPy Interoperability
PyTorch seamlessly integrates with NumPy.
* Tensor to NumPy: `b = a.numpy()`
* NumPy to Tensor: `d = torch.from_numpy(c)`
