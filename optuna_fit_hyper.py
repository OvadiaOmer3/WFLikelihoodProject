import optuna
import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import cross_val_score
import os
import sys
import pickle

import datetime

from generateSchechterFunctionSamples import schechter_curr_equation

models = []

# Define the objective function
def objective(trial: optuna.Trial) -> float:
    # Define the search space

    params = dict(
        populations=trial.suggest_int("populations", 50, 200),
        ncyclesperiteration=trial.suggest_int("ncyclesperiteration", 50, 150),
        alpha=trial.suggest_float("alpha", 0.05, 0.2),
        # fraction_replaced=0.01,
        # fraction_replaced_hof=0.005,
        population_size=trial.suggest_int("population_size", 50, 200),
        # parsimony=1e-4,
        topn=trial.suggest_int("topn", 5, 20),
        # weight_add_node=1.0,
        # weight_insert_node=3.0,
        # weight_delete_node=3.0,
        # weight_do_nothing=1.0,
        # weight_mutate_constant=10.0,
        # weight_mutate_operator=1.0,
        # weight_swap_operands=1.0,
        # weight_randomize=1.0,
        # crossover_probability=0.01,
        # perturbation_factor=1.0,
        maxsize=trial.suggest_int("maxsize", 10, 50),
        # warmup_maxsize_by=0.0,
        # use_frequency=True,
        # optimizer_nrestarts=3.0,
        # optimize_probability=1.0,
        # optimizer_iterations=10.0,
        tournament_selection_p=trial.suggest_float("tournament_selection_p", 0.9, 0.999),
        binary_operators=["+", "*", "/", "^"],
        unary_operators=["exp", "log", "sqrt", "sin"],
        constraints={'^': (-1, 1)},
        nested_constraints={
            "exp": {"exp": 0, "log": 0, "sqrt": 0},
            "log": {"exp": 0, "log": 0, "sqrt": 0},
            "sqrt": {"exp": 0, "log": 0, "sqrt": 0},
        }
    )

    
    # timeout_in_seconds = trial.suggest_int("timeout_in_seconds", 1 * 60, 8 * 60)
    # population_size = trial.suggest_int("population_size", 27, 200)
    # procs = trial.suggest_int("procs", 1, 8)
    
    # Initialize and fit the model
    model = PySRRegressor(
        **params,
        # timeout_in_seconds=timeout_in_seconds,
        # population_size=population_size,
        # procs=procs,
        verbosity=0,
        temp_equation_file=True,
        delete_tempfiles=False,
    )
    
    # Example data
    X = np.linspace(10**8, 10**12, 100)
    X = X.reshape(-1, 1)
    y = schechter_curr_equation(X)
    
    # Ensure `y` is a 2D array
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    model.fit(X, y)
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=5)
    
    models.append(model)

    # Return the negative mean score as the objective value
    return -np.mean(scores)

# Create a study and optimize the objective function
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1, show_progress_bar=True)

# Print the best trial
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Save the best model
best_model = models[study.best_trial.number]
pickle.dumps(best_model, open("best_model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".pkl", "wb"))