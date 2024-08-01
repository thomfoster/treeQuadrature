import pandas as pd
import matplotlib.pyplot as plt
import glob

from treeQuadrature.visualisation import plot_errors

# Read all CSV files and concatenate them into a single DataFrame
csv_files = glob.glob("test_results/kdSplit_mean_median_compare.csv")  # Update the path to your CSV files
all_data = pd.concat([pd.read_csv(f) for f in csv_files])

if __name__ == '__main__':
    plot_errors(all_data, 'mean_median_kdSplit', ['Camel'], False, True)