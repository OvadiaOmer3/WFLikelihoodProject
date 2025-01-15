import pandas as pd
from pysr import PySRRegressor
import numpy as np
import sympy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import datetime
import argparse
import pickle
from sklearn.model_selection import GridSearchCV

from TestPySRRegressor import create_fit_report, calculate_MAD
from generateSchechterFunctionSamples import schechter_curr_equation

KNOWN_EQUATIONS = {
    "rydberg": lambda x: (1.09677576 * 10**7) * (1/(x**2) - 1/(4**2)),
    "bode": lambda x: 0.4 + 0.3 * (2**x),
    "schechter": schechter_curr_equation,
}

def load_data(data_path):
    df = pd.read_csv(data_path)
    X = df['X'].values
    y = df['Y'].values
    return X, y

def learn_1d_x(X,y, out_path):
    model = PySRRegressor(
        niterations=1000,
        # populations=200,
        binary_operators=["+", "*", "/", "^"],
        unary_operators=["exp", "log", "sqrt", "sin"],
        procs=4,
        variable_names=["x1"],
        constraints={'^': (-1, 1)},
        nested_constraints={
            "exp": {"exp": 0, "log": 0, "sqrt": 0},
            "log": {"exp": 0, "log": 0, "sqrt": 0},
            "sqrt": {"exp": 0, "log": 0, "sqrt": 0},
        },
        # elementwise_loss="HuberLoss({})".format(1.5 * calculate_MAD(y)),
        # elementwise_loss="HuberLoss({})".format(1.5 * calculate_MAD(y)),
        # elementwise_loss="L1DistLoss()",
        temp_equation_file=True,
        delete_tempfiles=False,
        # verbosity=0,

    )

    param_grid = {
        "elementwise_loss": ["HuberLoss({})".format(1.5 * calculate_MAD(y)),
                             "L1DistLoss()", 
                             "LogitDistLoss()",
                             "L2DistLoss()",
                            ]
                             
    }

    # grid_search = GridSearchCV(
    #     model,
    #     param_grid,
    #     cv=5,
    #     n_jobs=4,
    #     verbose=1,
    #     scoring="neg_mean_squared_error"
    #     )
    
    # grid_search.fit(X, y)
    # model = grid_search.best_estimator_

    model.fit(X, y)
    

    y_test = y
    X_test = X


    # # Find the best equation by loss
    # best_idx = model.equations_.query(
    #         f"loss < {2 * model.equations_.loss.min()}"
    #     ).score.idxmax()
    
    # Find the best equation by score
    best_idx = model.equations_.score.idxmax()

    print("\n")
    print("Best equation (by score):")
    print(model.sympy(best_idx))
    print("Loss: {}".format(model.equations_.loss.min()))
    print("Avg. Loss: {}\n".format(model.equations_.loss.min()/len(y_test)))

    # Save the model
    pickle.dump(model, open(os.path.join(out_path, "model.pkl"), "wb"))

    return model

def plot_1d_x(X, y, model, out_path, known_equation=None, is_log=False): 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Plot the actual line
    ax.set_yscale('log')
    ax.set_xscale('log')
    if known_equation != None:
        if is_log:
            x_actual = np.logspace(np.log10(X.min()), np.log10(X.max()), 1000)
            # ax.set_xscale('log')   
        else:           
            x_actual = np.linspace(X.min(), X.max(), 1000)
        y_actual = known_equation(x_actual)
        ax.plot(x_actual, y_actual, label="Real", color="blue")    
    
    # Plot the model
    y_pred = model.predict(X)
    ax.scatter(X, y_pred, label="Model", color="red")

    # Plot the data
    ax.scatter(X, y, label="Data", color="black", marker="+", s=50)

    # Save the plot
    ax.legend()
    fig.savefig(os.path.join(out_path, "plot.png"))
    plt.close()

def main(data_path, is_log=False, known_equation=None):
    # Load the data
    X, y = load_data(data_path)

    X = X.reshape(-1, 1)

    # Split the data into training and testing sets

    # Learn the model
    out_path = data_path.replace("Samples", "Results").replace(".csv", "") + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_path, exist_ok=True)
    model = learn_1d_x(X, y, out_path)

    # Plot the model
    plot_1d_x(X, y, model, out_path, known_equation=known_equation, is_log=is_log)    

    report = create_fit_report(model, X, y)
    with open(os.path.join(out_path, "report.txt"), "w") as f:
        f.write(report)

    model.equations_.to_csv(os.path.join(out_path, "equations" + ".csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./Samples/rydberg_samples_consistant_X.csv")
    parser.add_argument("--is_log", action="store_true")
    args = parser.parse_args()

    main(args.data_path, is_log=args.is_log, known_equation=KNOWN_EQUATIONS.get(os.path.basename(args.data_path).split("_")[0]))
    # main("./Samples/schechter_samples.csv", is_log=True, known_equation=KNOWN_EQUATIONS.get("schechter"))