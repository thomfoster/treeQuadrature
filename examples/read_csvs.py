import pandas as pd
import glob

from treeQuadrature.visualisation import plot_errors, plot_times

# Read all CSV files and concatenate them into a single DataFrame
csv_files = glob.glob("test_results/sse_split/results_*csv")  # Update the path to your CSV files
all_data = pd.concat([pd.read_csv(f) for f in csv_files])

# genres = ['Camel', 'QuadCamel', 'SimpleGaussian', 'ExponentialProduct', 
#           'Quadratic', 'Ripple', 'Oscillatory', 'ProductPeak', 'C0function', 'CornerPeak', 'Discontinuous']
genres = ['Camel', 'Ripple']

# integrators = ['TQ with mean', 'TQ with Rbf', 'ActiveTQ', 'Vegas']

if __name__ == '__main__':
    plot_times(all_data, genres, font_size=17, filename_prefix='sse_split/',
               title=False, same_figure=True)
    plot_errors(all_data, genres, plot_title=False, error_bar=True, font_size=20, offset=0.1, 
                y_lim=[-200, 80], filename_prefix='sse_split/')