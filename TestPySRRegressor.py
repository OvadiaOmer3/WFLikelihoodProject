import pandas as pd
from pysr import PySRRegressor
import numpy as np
import sympy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
import os
import datetime
import argparse


def generate_2features_X_by_ranges(n_samples, x_1_range, x_2_range):
    x_1 = np.random.uniform(x_1_range[0], x_1_range[1], n_samples)
    x_2 = np.random.uniform(x_2_range[0], x_2_range[1], n_samples)
    X = np.column_stack((x_1, x_2))
    return X


def create_3d_graph(X, y, y_prediction, filename, should_show=False, should_save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate data ranges
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    z_min, z_max = min(y.min(), y_prediction.min()), max(y.max(), y_prediction.max())
    
    # Set plot limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    ax.scatter(X[:, 0], X[:, 1], y, color="blue")
    ax.scatter(X[:, 0], X[:, 1], y_prediction, color="red")

    if should_show:
        plt.show()

    if should_save:
        def update(i):
            ax.view_init(elev=10., azim=i)

        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), interval=200)
        ani.save(filename, writer='imagemagick', fps=15)

"""
Create bitmap.
X: 2D array of shape (n_samples, 2)
y: 1D array of shape (n_samples)

DO NOT Create 3d plot, instead express the depth of y as a color in a 2D plot.
"""
def create_bitmap(X, y, y_prediction, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Calculate data ranges
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    # Set plot limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Create a scatter plot with color representing the depth of y
    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.colorbar(sc)

    plt.savefig(filename)

    
def calculate_MAD(y_true):
    return np.mean(np.abs(y_true - np.mean(y_true)))    


def learn_2d_x(X, y, should_show=True, should_save_gif=False, filename="", should_dump_to_file=False, out_path=""):
    model = PySRRegressor(
        niterations=1000,
        binary_operators=["+", "*", "/"],
        unary_operators=["exp", "log", "sqrt"],
        procs=4,
        variable_names=["x1", "x2"],
        nested_constraints={
            "exp": {"exp": 0, "log": 0, "sqrt": 0},
            "log": {"exp": 0, "log": 0, "sqrt": 0},
            "sqrt": {"exp": 0, "log": 0, "sqrt": 0},
        },
        # elementwise_loss="HuberLoss({})".format(1.5 * calculate_MAD(y)),
        elementwise_loss="HuberLoss(50)",
        temp_equation_file=True,
        delete_tempfiles=False,

    )

    y_test = y
    X_test = X

    model.fit(X, y)

    # Find the best equation
    best_idx = model.equations_.query(
            f"loss < {2 * model.equations_.loss.min()}"
        ).score.idxmax()
    print("\n")
    print("Best equation:")
    print(model.sympy(best_idx))
    print("Loss: {}".format(model.equations_.loss.min()))
    print("Avg. Loss: {}\n".format(model.equations_.loss.min()/len(y_test)))

    # Plot the best equation (3d plot) 
    y_prediction = model.predict(X_test, index=best_idx)
    # if should_show:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color="blue")
    #     ax.scatter(X_test[:, 0], X_test[:, 1], y_prediction, color="red")
    #     plt.show()

    if should_save_gif or should_show:
        filename = "out.gif" if filename=="" else filename
        filename_no_ext, ext = os.path.splitext(filename)
        filename = filename_no_ext + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".gif"
        create_3d_graph(X_test, y_test, y_prediction, filename=os.path.join(out_path, filename), should_show=should_show, should_save=should_save_gif)
    
    if should_dump_to_file:
        # dump X_test, y_test, y_prediction to a file
        np.save(os.path.join(out_path, "X_test.npy"), X_test)
        np.save(os.path.join(out_path, "y_test.npy"), y_test)
        np.save(os.path.join(out_path, "y_prediction.npy"), y_prediction)
        print("Dumped X_test, y_test, y_prediction to files")
    
    return model, X_test, y_test, y_prediction



"""
Load data from pandas df csv.
df first column: x1
df second column: x2
df values: y
"""
def load_data_from_csv(filename):
    df = pd.read_csv(filename)
    # reshape so that X is (rows * columns)X2 and y is 1D (y = df[x1, x2])
    melted = pd.melt(df, id_vars=df.columns[0], var_name='x_1', value_name='f(x_1,x_2)')

    # Rename the index column to 'x_2'
    melted.rename(columns={df.columns[0]: 'x_2'}, inplace=True)

    # Extract numpy arrays
    x_combinations = melted[['x_1', 'x_2']].to_numpy()  # Shape: [num_combinations, 2]
    f_combinations = melted['f(x_1,x_2)'].to_numpy()    # Shape: [num_combinations]

    return x_combinations, f_combinations

def load_results_from_csv(path=""):
    X_test = np.load(os.path.join(path, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(path,"y_test.npy"),  allow_pickle=True)
    y_prediction = np.load(os.path.join(path, "y_prediction.npy"), allow_pickle=True)

    convert_to_float = lambda x: float(x)
    X_test = np.array([list(map(convert_to_float, x)) for x in X_test])
    y_test = np.array([float(x) for x in y_test])
    y_prediction = np.array([float(x) for x in y_prediction])

    return X_test, y_test, y_prediction


def load_data_from_csv_small_rec(filename):
    df = pd.read_csv(filename)

    df_sliced = df.iloc[40:63,33:79]
    df_x_column = df.iloc[40:63, 0]
    # add x_column as first column
    df = pd.concat([df_x_column, df_sliced], axis=1)

    
    df = df.reset_index(drop=True)

    # reshape so that X is (rows * columns)X2 and y is 1D (y = df[x1, x2])
    melted = pd.melt(df, id_vars=df.columns[0], var_name='x_1', value_name='f(x_1,x_2)')

    # Rename the index column to 'x_2'
    melted.rename(columns={df.columns[0]: 'x_2'}, inplace=True)

    # Extract numpy arrays
    x_combinations = melted[['x_1', 'x_2']].to_numpy()  # Shape: [num_combinations, 2]
    f_combinations = melted['f(x_1,x_2)'].to_numpy()    # Shape: [num_combinations]

    return x_combinations, f_combinations

def create_fit_report(model, X, y):
    # Find the best equation
    best_idx = model.equations_.query(
            f"loss < {2 * model.equations_.loss.min()}"
        ).score.idxmax()
    report = """
Best equation (IDX: {}):
{}

Sympy format:
{}

Loss: {}
Avg. Loss: {}

    """.format(
        best_idx,
        model.equations_.equation[best_idx],
        model.sympy(best_idx),
        model.equations_.loss.min(),
        model.equations_.loss.min()/len(y),
    )

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Symbolic Regression using PySR")
    parser.add_argument("--filename", type=str, default="loglik_grid_-1_-5.csv", help="CSV file to load data from")
    parser.add_argument("--should_show", action="store_true", help="Show 3D plot of the best equation")
    parser.add_argument("--should_save_gif", action="store_true", help="Save 3D plot of the best equation to a gif")
    parser.add_argument("--should_dump_to_file", action="store_true", help="Dump X_test, y_test, y_prediction to files")
    parser.add_argument("--output_filename", type=str, default="", help="Output filename for the gif")
    parser.add_argument("--should_plot_old_results", action="store_true", help="Plot old results from files")
    parser.add_argument("--old_results_path", type=str, default="", help="Path to old results files")
    args = parser.parse_args()
    return args


def main():
    # filename = "loglik_grid_-1_-5_SLICED.csv"
    # X, y = load_data_from_csv_small_rec(filename)
    args = parse_args()

    filename = args.filename
    if args.should_plot_old_results:
        fix_3d_graph(old_results_path=args.old_results_path, should_show=args.should_show, should_save=args.should_save_gif)
        return
    else:
        X, y = load_data_from_csv(filename)
        # model, X_test, y_test, y_prediction = learn_2d_x(X,
        #                                              y,
        #                                              should_show=False,
        #                                              should_save_gif=True,
        #                                              filename="loglik_grid_-1_-5.gif",
        #                                              should_dump_to_file=True,
        #                                              )
        out_path, ext = os.path.splitext(filename)
        out_path = "out" if out_path=="" else out_path
        test_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_path + "_" + test_time
        
        os.makedirs(out_path, exist_ok=False)

        model, X_test, y_test, y_prediction = learn_2d_x(X,
                                                        y,
                                                        should_show=args.should_show,
                                                        should_save_gif=args.should_save_gif,
                                                        filename=filename if args.output_filename=="" else args.output_filename,
                                                        should_dump_to_file=args.should_dump_to_file,
                                                        out_path=out_path
                                                        )
        
        report = create_fit_report(model, X_test, y_test)

        report_filename = os.path.join(out_path ,"REPORT_" + test_time + ".txt")
        with open(report_filename, "w") as f:
            f.write(report)

        model.equations_.to_csv(os.path.join(out_path, "equations_" + test_time + ".csv"))


def fix_3d_graph(old_results_path="", should_show=False, should_save=True):
    X_test, y_test, y_prediction = load_results_from_csv(old_results_path)
    filename = old_results_path + ".gif"
    filename_no_ext, ext = os.path.splitext(filename)
    filename = filename_no_ext + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".gif"
    create_3d_graph(X_test, y_test, y_prediction, filename=filename, should_show=should_show, should_save=should_save)
    # filename = filename_no_ext + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    # create_bitmap(X_test, y_test, y_prediction, filename=filename)


if __name__ == "__main__":
    # test_normal_dist_by_2features()
    main()
    # fix_3d_graph()
