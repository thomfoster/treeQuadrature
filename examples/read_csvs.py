import pandas as pd
import glob

from treeQuadrature.visualisation import plot_errors, plot_times

# Read all CSV files and concatenate them into a single DataFrame
csv_files = glob.glob("test_results/results_*csv")  # Update the path to your CSV files
all_data = pd.concat([pd.read_csv(f) for f in csv_files])

genres = ['Camel', 'QuadCamel', 'SimpleGaussian']
integrators = ['TQ with RBF', 'TQ with RBF (mean)', 'SMC', 'BMC', 'LimitedSampleIntegrator']

if __name__ == '__main__':
    # plot_times(all_data, genres, font_size=13)
    plot_errors(all_data, genres, error_bar=True, font_size=20, offset=0.2, integrators=integrators, 
                y_lim=[-150, 150])