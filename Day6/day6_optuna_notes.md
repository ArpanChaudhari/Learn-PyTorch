# PyTorch Day 6 Notes: Hyperparameter Tuning with Optuna

When training neural networks, hyperparameters (like learning rate, number of layers, number of neurons, optimizer type) heavily dictate the model's success. Guessing these manually is inefficient. Today's focus was using **Optuna** to automate this search efficiently.

## 1. The Optuna `objective` Function
To use Optuna, you wrap your model definition, data loading, and training loop inside a single `objective(trial)` function. This function must return the metric you want to optimize (e.g., validation accuracy or validation loss).

```python
import optuna
import torch.nn as nn
import torch.optim as optim

def objective(trial):
    # 1. Suggest Hyperparameters dynamically
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    
    # You can even suggest architectural choices!
    n_units = trial.suggest_int("n_units", 32, 256)
    
    # 2. Build model using suggested params
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, n_units),
        nn.ReLU(),
        nn.Linear(n_units, 10)
    )
    
    # 3. Instantiate the optimizer based on the suggestion
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    # 4. Standard PyTorch Training Loop
    # ... (Train your model for a few epochs)
    
    # 5. Return the validation accuracy
    # ... (Evaluate model on validation set)
    return val_accuracy
```

## 2. Creating and Running a Study
Once the objective function is defined, you create an Optuna `study`. You must specify the `direction`—whether you want to `maximize` (e.g., accuracy) or `minimize` (e.g., loss) the returned value.

```python
study = optuna.create_study(direction="maximize")
# Run 50 trials
study.optimize(objective, n_trials=50)
```

## 3. Extracting the Best Results
After the trials are complete, you can easily access the best performing hyperparameters and train your final, production-ready model with them.

```python
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
```

## 4. Why Optuna?
* **Efficiency:** Unlike Grid Search (which is exhaustively slow) or Random Search, Optuna uses a sophisticated Bayesian optimization algorithm (TPE - Tree-structured Parzen Estimator) to intelligently guess the next set of hyperparameters based on past results.
* **Pruning:** Optuna can automatically stop unpromising trials early (if the model is clearly failing after 2 epochs, it kills the trial and moves on), saving massive amounts of compute time.
