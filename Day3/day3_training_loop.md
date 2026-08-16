# PyTorch Day 3 Notes: The Custom Training Loop

## 1. Preparing Data for PyTorch
Before passing data into a PyTorch model, it needs to be formatted as Tensors.
* **From Pandas to NumPy:** You can extract values using `.values` or simply pass the dataframe/series to numpy.
* **From NumPy to PyTorch:** Use `torch.from_numpy()`. This is highly efficient because the resulting tensor shares the same underlying memory as the NumPy array.
* **Type Casting:** Often, models expect `float32` for features and `float32` or `int64` for labels, so you might need to use `.to(torch.float32)`.

## 2. Building a Model from Scratch
Instead of using PyTorch's built-in layers, you built a logistic regression model entirely from scratch using raw tensors!
* **Weights:** Initialized randomly with `requires_grad=True`. 
  `self.weights = torch.rand(features, 1, requires_grad=True)`
* **Bias:** Initialized to zero with `requires_grad=True`.
* **Forward Pass:** The prediction is a matrix multiplication followed by a sigmoid activation:
  `z = torch.matmul(X, self.weights) + self.bias`
  `y_pred = torch.sigmoid(z)`

## 3. The Custom Training Loop
The fundamental training loop in PyTorch consists of 5 distinct steps. You successfully implemented this loop manually:

1. **Forward Pass:** Pass the training data through the model to get predictions.
   `y_pred = model.forward(X_train_tensor)`
2. **Calculate Loss:** Compare predictions against true labels using a loss function (like Binary Cross Entropy).
   `loss = model.loss_function(y_pred, y_train_tensor)`
3. **Backward Pass:** Calculate the gradients of the loss with respect to all parameters (`weights` and `bias`).
   `loss.backward()`
4. **Parameter Update (Gradient Descent):** Adjust the weights in the opposite direction of the gradient to minimize the loss. **Crucial:** This must be done inside a `torch.no_grad()` block so PyTorch doesn't try to track the gradients of the update step itself!
   ```python
   with torch.no_grad():
       model.weights -= learning_rate * model.weights.grad
       model.bias -= learning_rate * model.bias.grad
   ```
5. **Zero Gradients:** Clear the gradients so they don't accumulate in the next epoch.
   `model.weights.grad.zero_()`

## 4. Evaluation
When evaluating the model on test data, it is important to wrap the forward pass in a `with torch.no_grad():` block. This prevents PyTorch from building a computation graph, saving significant memory and speeding up inference.
