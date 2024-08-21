import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import seaborn as sns

def plot_errors_from_csvs(csv_folder: str, plot_title: str = None, 
                          font_size: int = 12, output_folder: str = 'plots'):
    """
    Iterate through all CSV files in the specified folder and plot how 'error' 
    changes across various 'base_N' and 'P' parameter combinations using heatmaps.

    Parameters
    ----------
    csv_folder : str
        The folder containing the CSV files to process.
    plot_title : str, optional
        The title for the plots. If None, no title will be added.
    font_size : int, optional
        The font size for the plot labels and title. Default is 12.
    output_folder : str, optional
        The folder to save the generated plots. Default is 'plots'.

    Returns
    -------
    None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all CSV files in the specified folder
    csv_files = glob.glob(os.path.join(csv_folder, '*.csv'))
    for csv_file in csv_files:
        # Read the CSV file
        data = pd.read_csv(csv_file)

        # Convert columns to numeric, removing the "%" and converting to float
        data['error'] = pd.to_numeric(data['error'].str.rstrip('%').astype(float) / 100, errors='coerce')
        data['error_std'] = pd.to_numeric(data['error_std'].str.rstrip('%').astype(float) / 100, errors='coerce')
        data['base_N'] = pd.to_numeric(data['base_N'], errors='coerce')
        data['P'] = pd.to_numeric(data['P'], errors='coerce')

        # Pivot the data for heatmap
        error_pivot = data.pivot(index="P", columns="base_N", values="error")
        error_std_pivot = data.pivot(index="P", columns="base_N", values="error_std")

        plt.figure(figsize=(12, 8))
        
        # Create a heatmap for the error values
        sns.heatmap(error_pivot, annot=True, fmt=".2%", cmap="coolwarm", cbar_kws={'label': 'Error (%)'}, center=0)
        plt.title(f"{plot_title or 'Error Heatmap'} - {os.path.basename(csv_file)}", fontsize=font_size)
        plt.xlabel("base_N", fontsize=font_size)
        plt.ylabel("P", fontsize=font_size)

        # Save the plot
        plot_filename = os.path.join(output_folder, 
                                     f"{os.path.basename(csv_file).replace('.csv', '')}_heatmap.png")
        plt.savefig(plot_filename)
        plt.close()

        # Optionally, create a contour plot for error_std
        plt.figure(figsize=(12, 8))
        X, Y = np.meshgrid(sorted(data['base_N'].unique()), sorted(data['P'].unique()))
        Z = error_std_pivot.values
        plt.contourf(X, Y, Z, levels=20, cmap="coolwarm")
        plt.colorbar(label="Error Std (%)")
        plt.title(f"{plot_title or 'Error Std Contour'} - {os.path.basename(csv_file)}", 
                  fontsize=font_size)
        plt.xlabel("base_N", fontsize=font_size)
        plt.ylabel("P", fontsize=font_size)

        # Save the contour plot
        contour_filename = os.path.join(output_folder, 
                                        f"{os.path.basename(csv_file).replace('.csv', '')}_contour.png")
        plt.savefig(contour_filename)
        plt.close()

    print(f"Plots saved to {output_folder}")

if __name__ == '__main__':
    plot_errors_from_csvs(csv_folder='test_results/tree_params/', 
                        font_size=20, output_folder='figures/')