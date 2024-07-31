import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# Read all CSV files and concatenate them into a single DataFrame
csv_files = glob.glob("results/*.csv")  # Update the path to your CSV files
all_data = pd.concat([pd.read_csv(f) for f in csv_files])

# Extract the dimension information from the problem column and create a new column 'Dimension'
all_data['Dimension'] = all_data['problem'].str.extract(r'D=(\d+)').astype(int)

# Define the problem genres
genres = ["SimpleGaussian", "Camel", "QuadCamel"]

# Plotting the error and error_std for each genre and each algorithm
for genre in genres:
    genre_data = all_data[all_data['problem'].str.contains(genre)]
    dimensions = genre_data['Dimension'].unique()
    dimensions.sort()

    plt.figure(figsize=(14, 7))
    
    for integrator in genre_data['integrator'].unique():
        genre_integrator_data = genre_data[genre_data['integrator'] == integrator]
        errors = []
        error_stds = []
        
        for dim in dimensions:
            data = genre_integrator_data[genre_integrator_data['Dimension'] == dim]
            errors.append(data['error'].values[0])
            error_stds.append(data['error_std'].values[0])
        
        plt.errorbar(dimensions, errors, yerr=error_stds, label=integrator, capsize=5)
    
    plt.title(f'Error and Error Std for {genre}')
    plt.xlabel('Dimension')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.grid(True)
    plt.show()
