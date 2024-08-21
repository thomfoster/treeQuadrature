import pandas as pd
import glob

from treeQuadrature.visualisation import plot_errors, plot_times

# Read all CSV files and concatenate them into a single DataFrame
csv_files = glob.glob("test_results/fourth_run/results_*csv")  # Update the path to your CSV files
all_data = pd.concat([pd.read_csv(f) for f in csv_files])

genres = ['Camel', 'QuadCamel', 'SimpleGaussian', 'ExponentialProductProblem', 
          'QuadraticProblem', 'Ripple', 'Oscillatory', 'ProductPeak', 'C0function', 'CornerPeak', 'Discontinuous']
genres = ['C0function']
integrators = ['TQ with mean', 'ActiveTQ', 'TQ with Rbf','SMC', 'Vegas']

if __name__ == '__main__':
    # plot_times(all_data, genres, font_size=17, filename_prefix='fourth_run/', integrators=integrators, 
    #            title=False)
    plot_errors(all_data, genres, plot_title=False, error_bar=True, font_size=20, offset=0.05, integrators=integrators, 
                y_lim=[-15, 150], filename_prefix='fourth_run/')