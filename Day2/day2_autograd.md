# PyTorch Day 2 Notes: Autograd (Automatic Differentiation)

## 1. What is Autograd?
`autograd` is PyTorch's automatic differentiation engine. It powers neural network training by automatically calculating the gradients (derivatives) of your tensors, saving you from doing complex calculus by hand.

## 2. Tracking Gradients
By default, PyTorch tensors do **not** track gradients to save memory. 
* **Enable tracking:** Set `requires_grad=True` when creating the tensor: 
  `x = torch.tensor(3.0, requires_grad=True)`
* When a tensor has `requires_grad=True`, PyTorch tracks every operation applied to it and builds a **Computation Graph** in the background.

## 3. Calculating Gradients (`backward()`)
To calculate the gradient of a loss function with respect to your inputs:
1. Perform your forward pass (e.g., `y = x ** 2`)
2. Call `.backward()` on the final output tensor (e.g., `y.backward()`)
3. The gradient is now stored in the `.grad` attribute of the input tensor (e.g., `x.grad`).

*Example:* If $y = x^2$ and $x = 3.0$, then $\frac{dy}{dx} = 2x = 6.0$. After calling `y.backward()`, `x.grad` will be `6.0`.

## 4. The Computation Graph
When you perform operations on tensors with `requires_grad=True`, PyTorch builds a Directed Acyclic Graph (DAG). 
* Leaves: Input tensors created by the user.
* Roots: Output tensors.
* Edges/Nodes: Operations applied (`grad_fn` like `<PowBackward0>`, `<AddBackward0>`).

Because the graph is *dynamic* (Define-by-Run), it is recreated from scratch every single time you perform a forward pass. This is what allows for complex architectures like varying sequence lengths in NLP.

## 5. The Golden Rule: Clearing Gradients
* **The Problem:** PyTorch *accumulates* (adds) gradients by design. If you run `.backward()` multiple times in a loop without clearing them, the new gradients will be added to the old ones, leading to incorrect updates.
* **The Solution:** You must explicitly zero out the gradients before the next forward pass.
  * For a single tensor: `x.grad.zero_()`
  * For an optimizer (used in training loops): `optimizer.zero_grad()`

## 6. Disabling Gradient Tracking
You only need gradients during **Training**. When you are evaluating your model, making predictions, or deploying it, you should turn off gradient tracking to save memory and speed up computation.

There are 3 ways to do this:
1. `x.requires_grad_(False)` (In-place modification)
2. `x.detach()` (Creates a new tensor that does not share the computation graph)
3. `with torch.no_grad():` (The most common way. Wraps a block of code to disable tracking globally for that block).

```python
with torch.no_grad():
    y_pred = model(x_test) # No gradients will be tracked here
```
