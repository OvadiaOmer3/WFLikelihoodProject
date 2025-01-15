import numpy as np
import pandas as pd
import argparse

# Example constants
L_star = 1e10  # Characteristic luminosity in solar units
phi_star = 1e-3  # Normalization (number density)
alpha = -1.3  # Power-law slope

def schechter_function(L, L_star, phi_star, alpha):
    return phi_star * (L / L_star)**alpha * np.exp(-L / L_star)


def schechter_curr_equation(L):
    return schechter_function(L, L_star, phi_star, alpha)

# Generate samples
def main(X, data_path):
    Y = schechter_function(X, L_star, phi_star, alpha)
    df = pd.DataFrame({'X': X, 'Y': Y})
    df.to_csv(data_path, index=False)

if __name__ == '__main__':
    # X = np.logspace(8, 12, 100) 
    parser = argparse.ArgumentParser(description='Generate Schechter function samples')
    parser.add_argument('--data_path', type=str, help='Path to save the generated samples')
    args = parser.parse_args()
    X= np.linspace(10**8, 10**12, 100)
    main(X, args.data_path)